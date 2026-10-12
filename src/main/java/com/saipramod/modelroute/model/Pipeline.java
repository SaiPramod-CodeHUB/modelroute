package com.saipramod.modelroute.model;

import com.saipramod.modelroute.data.TicketGenerator;
import com.saipramod.modelroute.data.TicketGenerator.Split;
import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.router.ConfidenceRouter;

import java.util.ArrayList;
import java.util.List;

/**
 * The full offline training pipeline:
 * <ol>
 *   <li>generate labeled tickets: train / validation / test with no duplicate texts across splits, plus a
 *       distribution-shift test set in phrasing never seen in training</li>
 *   <li>build the LARGE model (offline stand-in: char+word n-gram linear model + kNN ensemble; or Claude via Spring AI)</li>
 *   <li>let the large model label an unlabeled pool (the expensive step you pay for once)</li>
 *   <li>DISTILL: train the small model on the large model's soft labels + a small gold set</li>
 *   <li>calibrate the small model (temperature scaling) and pick the routing threshold on validation</li>
 * </ol>
 */
public final class Pipeline {

    public record Config(int goldTrain, int unlabeledPool, int validation, int test, long seed, double targetRatio) {
        public static Config defaults() {
            return new Config(1500, 6000, 1000, 2000, 42, 0.99);
        }
    }

    public record Trained(List<String> labels, List<Ticket> gold, List<Ticket> pool, List<Ticket> validation, List<Ticket> test,
                          List<Ticket> shiftedTest,
                          Classifier large, LinearClassifier smallGoldOnly, LinearClassifier smallDistilled,
                          ConfidenceRouter router, ConfidenceRouter.ThresholdChoice threshold, long trainingMillis) { }

    private Pipeline() { }

    public static Trained run(Config c, Classifier externalLarge) {
        long t0 = System.currentTimeMillis();
        TicketGenerator gen = new TicketGenerator();
        List<String> labels = gen.intents();
        List<Ticket> gold = gen.generate(Split.TRAIN, c.goldTrain(), c.seed(), 0.02);
        List<Ticket> pool = gen.generate(Split.TRAIN, c.unlabeledPool(), c.seed() + 1, 0.0);
        java.util.Set<String> seen = new java.util.HashSet<>();
        gold.forEach(t -> seen.add(t.text()));
        pool.forEach(t -> seen.add(t.text()));
        List<Ticket> val = gen.generateDisjoint(Split.VALIDATION, c.validation(), c.seed() + 2, seen);
        List<Ticket> test = gen.generateDisjoint(Split.TEST, c.test(), c.seed() + 3, seen);
        List<Ticket> shifted = gen.generateDisjoint(Split.SHIFTED_TEST, c.test() / 2, c.seed() + 4, seen);

        Classifier large = externalLarge != null ? externalLarge : offlineLarge(labels, gold, pool, c.seed());

        // The large model labels the unlabeled pool: soft targets keep its uncertainty.
        List<String> texts = new ArrayList<>();
        List<double[]> targets = new ArrayList<>();
        for (Ticket t : pool) {
            texts.add(t.text());
            targets.add(large.predict(t.text()).probs());
        }
        for (Ticket t : gold) {
            texts.add(t.text());
            double[] y = new double[labels.size()];
            y[labels.indexOf(t.intent())] = 1;
            targets.add(y);
        }
        LinearClassifier distilled = new LinearClassifier("small-distilled", smallHasher(), labels)
                .fitSoft(texts, targets, 6, c.seed());
        distilled.calibrate(val);

        LinearClassifier goldOnly = new LinearClassifier("small-gold-only", smallHasher(), labels).fitGold(gold, 6, c.seed());
        goldOnly.calibrate(val);

        List<Classifier.Prediction> sv = val.stream().map(t -> distilled.predict(t.text())).toList();
        List<Classifier.Prediction> lv = val.stream().map(t -> large.predict(t.text())).toList();
        ConfidenceRouter.ThresholdChoice choice = ConfidenceRouter.choose(val, sv, lv, c.targetRatio());
        ConfidenceRouter router = new ConfidenceRouter(distilled, large, choice.threshold());
        return new Trained(labels, gold, pool, val, test, shifted, large, goldOnly, distilled, router, choice,
                System.currentTimeMillis() - t0);
    }

    /**
     * Offline stand-in for a frontier model: heavier features and a kNN memory over every
     * labeled example. It sees gold labels for the whole training pool, playing the role of a
     * model that already "knows" the task, which a frontier LLM does zero-shot.
     */
    static Classifier offlineLarge(List<String> labels, List<Ticket> gold, List<Ticket> pool, long seed) {
        List<Ticket> all = new ArrayList<>(gold);
        all.addAll(pool);
        FeatureHasher rich = new FeatureHasher(1 << 20, 3, 2, 5);
        LinearClassifier lin = new LinearClassifier("large-linear", rich, labels).fitGold(all, 8, seed);
        KnnClassifier knn = new KnnClassifier(new FeatureHasher(1 << 18, 2, 3, 4), labels, 9).fit(all);
        return new EnsembleClassifier("large (offline stand-in)", List.of(lin, knn));
    }

    static FeatureHasher smallHasher() {
        return new FeatureHasher(1 << 15, 2, 0, 0);
    }
}
