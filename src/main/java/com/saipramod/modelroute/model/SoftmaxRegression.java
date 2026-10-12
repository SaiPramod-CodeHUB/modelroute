package com.saipramod.modelroute.model;

import com.saipramod.modelroute.model.FeatureHasher.SparseVec;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Multinomial logistic regression trained with SGD on sparse features. Supports hard
 * labels and soft targets (teacher probability distributions), which is what knowledge
 * distillation needs: the student learns the teacher's full belief, not just its argmax.
 */
public final class SoftmaxRegression {

    private final int dims;
    private final int classes;
    private final float[][] w;   // [class][dim]
    private final float[] b;

    public SoftmaxRegression(int dims, int classes) {
        this.dims = dims;
        this.classes = classes;
        this.w = new float[classes][dims];
        this.b = new float[classes];
    }

    /** targets[i] is a probability distribution over classes (one-hot for gold labels). */
    public void train(List<SparseVec> xs, List<double[]> targets, int epochs, double lr0, double l2, long seed) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < xs.size(); i++) {
            order.add(i);
        }
        Random r = new Random(seed);
        long step = 0;
        for (int ep = 0; ep < epochs; ep++) {
            Collections.shuffle(order, r);
            for (int i : order) {
                double lr = lr0 / (1 + 1e-4 * step++);
                SparseVec x = xs.get(i);
                double[] p = proba(x, 1.0);
                double[] t = targets.get(i);
                for (int c = 0; c < classes; c++) {
                    float g = (float) (p[c] - t[c]);
                    if (g == 0) {
                        continue;
                    }
                    float[] wc = w[c];
                    for (int k = 0; k < x.idx().length; k++) {
                        int j = x.idx()[k];
                        wc[j] -= (float) (lr * (g * x.val()[k] + l2 * wc[j]));
                    }
                    b[c] -= (float) (lr * g);
                }
            }
        }
    }

    public double[] logits(SparseVec x) {
        double[] z = new double[classes];
        for (int c = 0; c < classes; c++) {
            double s = b[c];
            float[] wc = w[c];
            for (int k = 0; k < x.idx().length; k++) {
                s += wc[x.idx()[k]] * x.val()[k];
            }
            z[c] = s;
        }
        return z;
    }

    /** Softmax with temperature (temperature scaling is how the router's confidence is calibrated). */
    public double[] proba(SparseVec x, double temperature) {
        return softmax(logits(x), temperature);
    }

    public static double[] softmax(double[] z, double t) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : z) {
            max = Math.max(max, v / t);
        }
        double sum = 0;
        double[] p = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            p[i] = Math.exp(z[i] / t - max);
            sum += p[i];
        }
        for (int i = 0; i < z.length; i++) {
            p[i] /= sum;
        }
        return p;
    }

    /** Number of non-zero parameters: a fair "model size" for a hashed linear model. */
    public long nonZeroParams() {
        long n = 0;
        for (float[] row : w) {
            for (float v : row) {
                if (v != 0) {
                    n++;
                }
            }
        }
        return n + classes;
    }

    public int dims() {
        return dims;
    }
}
