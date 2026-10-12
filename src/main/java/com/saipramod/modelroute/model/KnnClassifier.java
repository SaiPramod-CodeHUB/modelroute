package com.saipramod.modelroute.model;

import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.model.FeatureHasher.SparseVec;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/** Cosine k-nearest-neighbours over every training example: memory-heavy and slow per request, but strong. */
public final class KnnClassifier implements Classifier {

    private final FeatureHasher hasher;
    private final List<String> labels;
    private final int k;
    private final List<SparseVec> vecs = new ArrayList<>();
    private final List<Integer> ys = new ArrayList<>();

    public KnnClassifier(FeatureHasher hasher, List<String> labels, int k) {
        this.hasher = hasher;
        this.labels = labels;
        this.k = k;
    }

    public KnnClassifier fit(List<Ticket> data) {
        for (Ticket t : data) {
            vecs.add(hasher.vectorize(t.text()));
            ys.add(labels.indexOf(t.intent()));
        }
        return this;
    }

    @Override
    public Prediction predict(String text) {
        long t0 = System.nanoTime();
        SparseVec q = hasher.vectorize(text);
        float[] dense = new float[hasher.dims()];
        for (int i = 0; i < q.idx().length; i++) {
            dense[q.idx()[i]] = q.val()[i];
        }
        PriorityQueue<double[]> top = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        for (int d = 0; d < vecs.size(); d++) {
            SparseVec v = vecs.get(d);
            double s = 0;
            for (int i = 0; i < v.idx().length; i++) {
                s += dense[v.idx()[i]] * v.val()[i];
            }
            top.offer(new double[] {s, d});
            if (top.size() > k) {
                top.poll();
            }
        }
        double[] p = new double[labels.size()];
        double total = 0;
        for (double[] e : top) {
            double w = Math.max(0, e[0]);
            p[ys.get((int) e[1])] += w;
            total += w;
        }
        for (int i = 0; i < p.length; i++) {
            p[i] = total == 0 ? 1.0 / p.length : p[i] / total;
        }
        int best = Classifier.argmax(p);
        return new Prediction(labels.get(best), p[best], p, (System.nanoTime() - t0) / 1000, name());
    }

    @Override
    public List<String> labels() {
        return labels;
    }

    @Override
    public String name() {
        return "kNN(k=" + k + ")";
    }
}
