package com.saipramod.modelroute.model;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Hashing vectoriser: word n-grams and character n-grams hashed into a fixed number of
 * dimensions, log-scaled term frequency, L2-normalised. No vocabulary to store, so the
 * model file is just the weight matrix, and unseen words still map somewhere.
 */
public final class FeatureHasher {

    public record SparseVec(int[] idx, float[] val) { }

    private final int dims;
    private final int maxWordN;
    private final int minCharN;
    private final int maxCharN;

    public FeatureHasher(int dims, int maxWordN, int minCharN, int maxCharN) {
        this.dims = dims;
        this.maxWordN = maxWordN;
        this.minCharN = minCharN;
        this.maxCharN = maxCharN;
    }

    public SparseVec vectorize(String text) {
        String norm = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9$€#' ]", " ").replaceAll("\\s+", " ").strip();
        String[] words = norm.isEmpty() ? new String[0] : norm.split(" ");
        Map<Integer, Float> counts = new HashMap<>();
        for (int n = 1; n <= maxWordN; n++) {
            for (int i = 0; i + n <= words.length; i++) {
                StringBuilder sb = new StringBuilder("w").append(n).append(':');
                for (int j = i; j < i + n; j++) {
                    sb.append(words[j]).append(' ');
                }
                add(counts, sb.toString(), 1f);
            }
        }
        if (maxCharN > 0) {
            for (String w : words) {
                String p = "<" + w + ">";
                for (int n = minCharN; n <= maxCharN; n++) {
                    for (int i = 0; i + n <= p.length(); i++) {
                        add(counts, "c:" + p.substring(i, i + n), 0.5f);
                    }
                }
            }
        }
        int[] idx = new int[counts.size()];
        float[] val = new float[counts.size()];
        int k = 0;
        double norm2 = 0;
        for (var e : counts.entrySet()) {
            idx[k] = e.getKey();
            val[k] = (float) Math.log1p(e.getValue());
            norm2 += val[k] * val[k];
            k++;
        }
        float inv = norm2 == 0 ? 0 : (float) (1 / Math.sqrt(norm2));
        for (int i = 0; i < val.length; i++) {
            val[i] *= inv;
        }
        return new SparseVec(idx, val);
    }

    private void add(Map<Integer, Float> counts, String feature, float w) {
        int h = feature.hashCode() * 0x9E3779B1;
        counts.merge(Math.floorMod(h ^ (h >>> 16), dims), w, Float::sum);
    }

    public int dims() {
        return dims;
    }

    public String describe() {
        return "word 1-" + maxWordN + "grams" + (maxCharN > 0 ? " + char " + minCharN + "-" + maxCharN + "grams" : "")
                + ", 2^" + Integer.numberOfTrailingZeros(dims) + " dims";
    }
}
