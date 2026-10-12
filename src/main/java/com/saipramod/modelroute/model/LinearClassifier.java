package com.saipramod.modelroute.model;

import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.model.FeatureHasher.SparseVec;

import java.util.ArrayList;
import java.util.List;

/** Hasher + softmax regression, with a calibration temperature. Used for both the small student and as part of the large stand-in. */
public final class LinearClassifier implements Classifier {

    private final String name;
    private final FeatureHasher hasher;
    private final SoftmaxRegression model;
    private final List<String> labels;
    private double temperature = 1.0;

    public LinearClassifier(String name, FeatureHasher hasher, List<String> labels) {
        this.name = name;
        this.hasher = hasher;
        this.labels = labels;
        this.model = new SoftmaxRegression(hasher.dims(), labels.size());
    }

    public LinearClassifier fitGold(List<Ticket> data, int epochs, long seed) {
        List<double[]> targets = new ArrayList<>();
        for (Ticket t : data) {
            double[] y = new double[labels.size()];
            y[labels.indexOf(t.intent())] = 1;
            targets.add(y);
        }
        return fitSoft(data.stream().map(Ticket::text).toList(), targets, epochs, seed);
    }

    public LinearClassifier fitSoft(List<String> texts, List<double[]> targets, int epochs, long seed) {
        List<SparseVec> xs = texts.stream().map(hasher::vectorize).toList();
        model.train(xs, targets, epochs, 0.5, 1e-6, seed);
        return this;
    }

    /** Temperature scaling: pick T minimising negative log-likelihood on held-out data. */
    public double calibrate(List<Ticket> validation) {
        List<SparseVec> xs = validation.stream().map(t -> hasher.vectorize(t.text())).toList();
        List<double[]> logits = xs.stream().map(model::logits).toList();
        double bestT = 1;
        double bestNll = Double.MAX_VALUE;
        for (double t = 0.25; t <= 6.0; t += 0.05) {
            double nll = 0;
            for (int i = 0; i < logits.size(); i++) {
                double[] p = SoftmaxRegression.softmax(logits.get(i), t);
                nll -= Math.log(Math.max(1e-12, p[labels.indexOf(validation.get(i).intent())]));
            }
            if (nll < bestNll) {
                bestNll = nll;
                bestT = t;
            }
        }
        temperature = bestT;
        return bestT;
    }

    @Override
    public Prediction predict(String text) {
        long t0 = System.nanoTime();
        double[] p = model.proba(hasher.vectorize(text), temperature);
        int k = Classifier.argmax(p);
        return new Prediction(labels.get(k), p[k], p, (System.nanoTime() - t0) / 1000, name);
    }

    @Override
    public List<String> labels() {
        return labels;
    }

    @Override
    public String name() {
        return name;
    }

    public long parameters() {
        return model.nonZeroParams();
    }

    public double temperature() {
        return temperature;
    }

    public String features() {
        return hasher.describe();
    }
}
