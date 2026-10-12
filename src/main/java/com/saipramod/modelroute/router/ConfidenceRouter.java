package com.saipramod.modelroute.router;

import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.model.Classifier;
import com.saipramod.modelroute.model.Classifier.Prediction;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Cheap-first routing: the small model answers when its calibrated confidence clears the
 * threshold, otherwise the request falls back to the large model. The threshold is chosen
 * on validation data as the lowest fallback rate that keeps accuracy within a target share
 * of the large model's.
 */
public final class ConfidenceRouter implements Classifier {

    public record Routed(Prediction prediction, boolean fellBack, Prediction small) { }

    public record ThresholdChoice(double threshold, double fallbackRate, double routerAccuracy, double largeAccuracy) { }

    private final Classifier small;
    private final Classifier large;
    private volatile double threshold;
    private final AtomicLong total = new AtomicLong();
    private final AtomicLong fallbacks = new AtomicLong();

    public ConfidenceRouter(Classifier small, Classifier large, double threshold) {
        this.small = small;
        this.large = large;
        this.threshold = threshold;
    }

    public Routed route(String text) {
        total.incrementAndGet();
        Prediction s = small.predict(text);
        if (s.confidence() >= threshold) {
            return new Routed(s, false, s);
        }
        fallbacks.incrementAndGet();
        Prediction l = large.predict(text);
        return new Routed(new Prediction(l.label(), l.confidence(), l.probs(), l.latencyMicros() + s.latencyMicros(),
                l.model()), true, s);
    }

    @Override
    public Prediction predict(String text) {
        return route(text).prediction();
    }

    /**
     * @param targetRatio e.g. 0.99 = router accuracy must be at least 99% of the large model's
     */
    public static ThresholdChoice choose(List<Ticket> validation, List<Prediction> smallPreds, List<Prediction> largePreds,
                                         double targetRatio) {
        int n = validation.size();
        double largeAcc = 0;
        for (int i = 0; i < n; i++) {
            largeAcc += largePreds.get(i).label().equals(validation.get(i).intent()) ? 1 : 0;
        }
        largeAcc /= n;
        ThresholdChoice best = new ThresholdChoice(1.01, 1.0, largeAcc, largeAcc);
        for (double t = 0.30; t <= 0.995; t += 0.005) {
            int fb = 0;
            int correct = 0;
            for (int i = 0; i < n; i++) {
                boolean useLarge = smallPreds.get(i).confidence() < t;
                Prediction p = useLarge ? largePreds.get(i) : smallPreds.get(i);
                fb += useLarge ? 1 : 0;
                correct += p.label().equals(validation.get(i).intent()) ? 1 : 0;
            }
            double acc = (double) correct / n;
            double rate = (double) fb / n;
            if (acc >= targetRatio * largeAcc && rate < best.fallbackRate()) {
                best = new ThresholdChoice(Math.round(t * 1000) / 1000.0, rate, acc, largeAcc);
            }
        }
        return best;
    }

    public void setThreshold(double t) {
        this.threshold = t;
    }

    public double threshold() {
        return threshold;
    }

    public double fallbackRate() {
        long t = total.get();
        return t == 0 ? 0 : (double) fallbacks.get() / t;
    }

    public long requests() {
        return total.get();
    }

    @Override
    public List<String> labels() {
        return small.labels();
    }

    @Override
    public String name() {
        return "router(" + small.name() + " -> " + large.name() + ")";
    }
}
