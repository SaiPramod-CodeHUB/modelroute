package com.saipramod.modelroute.eval;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.model.Classifier;
import com.saipramod.modelroute.model.Classifier.Prediction;
import com.saipramod.modelroute.model.Pipeline;
import com.saipramod.modelroute.router.ConfidenceRouter;
import com.saipramod.modelroute.router.CostModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Compares the large model, the small model trained on gold labels only, the distilled small
 * model and the confidence router on a test set built from phrasings no model saw in
 * training. Reports accuracy, macro-F1, quality retained vs. the large model, fallback rate,
 * p50/p95 latency and cost per 1,000 requests.
 *
 * <pre>mvn -q compile exec:java</pre>
 */
public final class EvalRunner {

    /** Latency profile assumed for a hosted frontier API call (lognormal around the median). */
    static final double API_MEDIAN_MS = 700;
    static final double API_SIGMA = 0.45;

    public record Row(String system, double accuracy, double macroF1, double qualityRetained, double fallbackRate,
                      double p50Ms, double p95Ms, double costPer1k) { }

    public record SweepPoint(double threshold, double fallbackRate, double accuracy, double costPer1k) { }

    public record Report(int goldTrain, int pool, int validation, int test, String largeModel, String smallFeatures,
                         long smallParams, double temperature, ConfidenceRouter.ThresholdChoice threshold, List<Row> rows,
                         List<Row> shiftedRows, List<SweepPoint> sweep, List<SweepPoint> shiftedSweep,
                         Map<String, Double> perClassF1Small, double measuredLargeComputeP50Ms, long trainingMillis) { }

    private EvalRunner() { }

    public static void main(String[] args) throws IOException {
        Report r = run(Pipeline.Config.defaults(), null);
        Path out = Path.of("eval/results");
        Files.createDirectories(out);
        new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT).writeValue(out.resolve("results.json").toFile(), r);
        String md = markdown(r);
        Files.writeString(out.resolve("results.md"), md);
        System.out.println(md);
    }

    public static Report run(Pipeline.Config cfg, Classifier externalLarge) {
        Pipeline.Trained t = Pipeline.run(cfg, externalLarge);
        CostModel cost = CostModel.defaults();
        boolean simulateApi = externalLarge == null;
        List<Double> measuredLarge = new ArrayList<>();
        List<Row> rows = evaluate(t, t.test(), cost, new Random(cfg.seed()), simulateApi, measuredLarge);
        List<Row> shifted = evaluate(t, t.shiftedTest(), cost, new Random(cfg.seed() + 1), simulateApi, new ArrayList<>());
        measuredLarge.sort(null);
        return new Report(cfg.goldTrain(), cfg.unlabeledPool(), cfg.validation(), cfg.test(), t.large().name(),
                t.smallDistilled().features(), t.smallDistilled().parameters(), t.smallDistilled().temperature(),
                t.threshold(), rows, shifted, sweep(t, t.test(), cost), sweep(t, t.shiftedTest(), cost), perClassF1(t.test(), t.test().stream().map(x -> t.smallDistilled().predict(x.text())).toList(), t.labels()),
                r2(pct(measuredLarge, 50)), t.trainingMillis());
    }

    static List<Row> evaluate(Pipeline.Trained t, List<Ticket> test, CostModel cost, Random lat, boolean simulateApi,
                              List<Double> measuredLarge) {
        // Run every system on the same test tickets.
        List<Prediction> large = new ArrayList<>();
        List<Prediction> goldOnly = new ArrayList<>();
        List<Prediction> distilled = new ArrayList<>();
        List<Double> largeLatency = new ArrayList<>();
        for (Ticket tk : test) {
            Prediction l = t.large().predict(tk.text());
            large.add(l);
            measuredLarge.add(l.latencyMicros() / 1000.0);
            largeLatency.add(simulateApi ? Math.exp(Math.log(API_MEDIAN_MS) + API_SIGMA * lat.nextGaussian()) : l.latencyMicros() / 1000.0);
            goldOnly.add(t.smallGoldOnly().predict(tk.text()));
            distilled.add(t.smallDistilled().predict(tk.text()));
        }
        double largeAcc = accuracy(test, large);
        List<Row> rows = new ArrayList<>();

        List<Double> lLat = new ArrayList<>(largeLatency);
        double largeCost = test.stream().mapToDouble(tk -> cost.largeCost(tk.text())).sum();
        rows.add(row("Large model only", test, large, largeAcc, 1.0, lLat, largeCost));

        rows.add(smallRow("Small, gold labels only", test, goldOnly, largeAcc, cost));
        rows.add(smallRow("Small, distilled", test, distilled, largeAcc, cost));

        // Router: small first, large when not confident.
        double thr = t.threshold().threshold();
        List<Prediction> routed = new ArrayList<>();
        List<Double> rLat = new ArrayList<>();
        double rCost = 0;
        int fb = 0;
        for (int i = 0; i < test.size(); i++) {
            Prediction s = distilled.get(i);
            double ms = s.latencyMicros() / 1000.0;
            rCost += cost.smallCost(s.latencyMicros());
            if (s.confidence() < thr) {
                fb++;
                routed.add(large.get(i));
                ms += largeLatency.get(i);
                rCost += cost.largeCost(test.get(i).text());
            } else {
                routed.add(s);
            }
            rLat.add(ms);
        }
        rows.add(row("Router (distilled small -> large)", test, routed, largeAcc, (double) fb / test.size(), rLat, rCost));
        return rows;
    }

    /** Accuracy / fallback / cost trade-off for a range of thresholds: the curve an operator tunes on. */
    static List<SweepPoint> sweep(Pipeline.Trained t, List<Ticket> test, CostModel cost) {
        List<Prediction> small = test.stream().map(x -> t.smallDistilled().predict(x.text())).toList();
        List<Prediction> large = test.stream().map(x -> t.large().predict(x.text())).toList();
        List<SweepPoint> out = new ArrayList<>();
        for (double thr : new double[] {0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99}) {
            int fb = 0;
            int ok = 0;
            double c = 0;
            for (int i = 0; i < test.size(); i++) {
                boolean useLarge = small.get(i).confidence() < thr;
                Prediction p = useLarge ? large.get(i) : small.get(i);
                fb += useLarge ? 1 : 0;
                ok += p.label().equals(test.get(i).intent()) ? 1 : 0;
                c += cost.smallCost(small.get(i).latencyMicros()) + (useLarge ? cost.largeCost(test.get(i).text()) : 0);
            }
            out.add(new SweepPoint(thr, r4((double) fb / test.size()), r4((double) ok / test.size()), r4(c / test.size() * 1000)));
        }
        return out;
    }

    private static Row smallRow(String name, List<Ticket> test, List<Prediction> preds, double largeAcc, CostModel cost) {
        List<Double> lat = preds.stream().map(p -> p.latencyMicros() / 1000.0).toList();
        double c = preds.stream().mapToDouble(p -> cost.smallCost(p.latencyMicros())).sum();
        return row(name, test, preds, largeAcc, 0, new ArrayList<>(lat), c);
    }

    private static Row row(String name, List<Ticket> test, List<Prediction> preds, double largeAcc, double fallback,
                           List<Double> latencies, double totalCost) {
        double acc = accuracy(test, preds);
        latencies.sort(null);
        return new Row(name, r4(acc), r4(macroF1(test, preds)), r4(acc / largeAcc), r4(fallback),
                r2(pct(latencies, 50)), r2(pct(latencies, 95)), r4(totalCost / test.size() * 1000));
    }

    static double accuracy(List<Ticket> gold, List<Prediction> preds) {
        int c = 0;
        for (int i = 0; i < gold.size(); i++) {
            c += preds.get(i).label().equals(gold.get(i).intent()) ? 1 : 0;
        }
        return (double) c / gold.size();
    }

    static Map<String, Double> perClassF1(List<Ticket> gold, List<Prediction> preds, List<String> labels) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (String l : labels) {
            int tp = 0, fp = 0, fn = 0;
            for (int i = 0; i < gold.size(); i++) {
                boolean g = gold.get(i).intent().equals(l);
                boolean p = preds.get(i).label().equals(l);
                tp += g && p ? 1 : 0;
                fp += !g && p ? 1 : 0;
                fn += g && !p ? 1 : 0;
            }
            double prec = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
            double rec = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
            out.put(l, r4(prec + rec == 0 ? 0 : 2 * prec * rec / (prec + rec)));
        }
        return out;
    }

    static double macroF1(List<Ticket> gold, List<Prediction> preds) {
        List<String> labels = gold.stream().map(Ticket::intent).distinct().toList();
        return perClassF1(gold, preds, labels).values().stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }

    private static double pct(List<Double> sorted, double p) {
        if (sorted.isEmpty()) {
            return 0;
        }
        int i = (int) Math.ceil(p / 100 * sorted.size()) - 1;
        return sorted.get(Math.max(0, Math.min(sorted.size() - 1, i)));
    }

    private static double r2(double v) {
        return Math.round(v * 100) / 100.0;
    }

    private static double r4(double v) {
        return Math.round(v * 10000) / 10000.0;
    }

    static String markdown(Report r) {
        StringBuilder sb = new StringBuilder("## ModelRoute eval\n\n");
        sb.append(String.format(Locale.ROOT, "%d gold-labeled + %d large-model-labeled training tickets · %d validation · "
                        + "%d test tickets (no text overlap with training) · 12 intents%n%n",
                r.goldTrain(), r.pool(), r.validation(), r.test()));
        sb.append("| System | Accuracy | Macro-F1 | Quality vs large | Fallback rate | p50 latency | p95 latency | Cost / 1K requests |\n");
        sb.append("|---|---|---|---|---|---|---|---|\n");
        for (Row w : r.rows()) {
            sb.append(String.format(Locale.ROOT, "| %s | %.1f%% | %.3f | %.1f%% | %.1f%% | %.2f ms | %.2f ms | $%.4f |%n",
                    w.system(), w.accuracy() * 100, w.macroF1(), w.qualityRetained() * 100, w.fallbackRate() * 100,
                    w.p50Ms(), w.p95Ms(), w.costPer1k()));
        }
        Row large = r.rows().get(0);
        Row router = r.rows().get(r.rows().size() - 1);
        sb.append(String.format(Locale.ROOT, "%n**Router vs. large model only:** cost per 1K requests $%.4f → $%.4f, "
                        + "p95 latency %.0f ms → %.2f ms, quality retained %.1f%%.%n",
                large.costPer1k(), router.costPer1k(), large.p95Ms(), router.p95Ms(), router.qualityRetained() * 100));
        sb.append(String.format(Locale.ROOT, "%nRouting threshold %.3f on calibrated confidence (temperature %.2f), chosen on validation "
                        + "to keep ≥ 99%% of the large model's accuracy (validation: router %.1f%% vs large %.1f%%, fallback %.1f%%).%n",
                r.threshold().threshold(), r.temperature(), r.threshold().routerAccuracy() * 100,
                r.threshold().largeAccuracy() * 100, r.threshold().fallbackRate() * 100));
        sb.append(String.format(Locale.ROOT, "%nSmall model: %s, %,d non-zero parameters. Large model: %s "
                        + "(measured compute p50 %.2f ms; latency above uses a simulated hosted-API profile, median %.0f ms).%n",
                r.smallFeatures(), r.smallParams(), r.largeModel(), r.measuredLargeComputeP50Ms(), API_MEDIAN_MS));
        sb.append(String.format(Locale.ROOT, "%n**Distribution shift** (%d tickets in casual phrasing never seen in training):%n%n",
                r.test() / 2));
        sb.append("| System | Accuracy | Macro-F1 | Fallback rate |\n|---|---|---|---|\n");
        for (Row w : r.shiftedRows()) {
            sb.append(String.format(Locale.ROOT, "| %s | %.1f%% | %.3f | %.1f%% |%n", w.system(), w.accuracy() * 100,
                    w.macroF1(), w.fallbackRate() * 100));
        }
        sb.append("\nThreshold sweep (what an operator tunes): fallback rate / accuracy / cost per 1K\n\n");
        sb.append("| Threshold | In-distribution | Shifted |\n|---|---|---|\n");
        for (int i = 0; i < r.sweep().size(); i++) {
            SweepPoint a = r.sweep().get(i);
            SweepPoint b = r.shiftedSweep().get(i);
            sb.append(String.format(Locale.ROOT, "| %.2f | %.0f%% / %.1f%% / $%.3f | %.0f%% / %.1f%% / $%.3f |%n", a.threshold(),
                    a.fallbackRate() * 100, a.accuracy() * 100, a.costPer1k(), b.fallbackRate() * 100, b.accuracy() * 100, b.costPer1k()));
        }
        sb.append("\nPer-intent F1, distilled small model:\n\n| Intent | F1 |\n|---|---|\n");
        r.perClassF1Small().forEach((k, v) -> sb.append(String.format(Locale.ROOT, "| %s | %.3f |%n", k, v)));
        return sb.toString();
    }
}
