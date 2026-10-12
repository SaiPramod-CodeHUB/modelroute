package com.saipramod.modelroute.model;

import java.util.List;

/** Averages member probability distributions. Offline stand-in for the "large" model. */
public final class EnsembleClassifier implements Classifier {

    private final String name;
    private final List<Classifier> members;

    public EnsembleClassifier(String name, List<Classifier> members) {
        this.name = name;
        this.members = members;
    }

    @Override
    public Prediction predict(String text) {
        long t0 = System.nanoTime();
        double[] p = null;
        for (Classifier c : members) {
            double[] q = c.predict(text).probs();
            if (p == null) {
                p = q.clone();
            } else {
                for (int i = 0; i < p.length; i++) {
                    p[i] += q[i];
                }
            }
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= members.size();
        }
        int k = Classifier.argmax(p);
        return new Prediction(labels().get(k), p[k], p, (System.nanoTime() - t0) / 1000, name);
    }

    @Override
    public List<String> labels() {
        return members.get(0).labels();
    }

    @Override
    public String name() {
        return name;
    }
}
