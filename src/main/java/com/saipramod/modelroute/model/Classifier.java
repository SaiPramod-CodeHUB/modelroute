package com.saipramod.modelroute.model;

import java.util.List;

/** Anything that maps a ticket to an intent with a probability distribution. */
public interface Classifier {

    record Prediction(String label, double confidence, double[] probs, long latencyMicros, String model) { }

    Prediction predict(String text);

    List<String> labels();

    String name();

    static int argmax(double[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }
}
