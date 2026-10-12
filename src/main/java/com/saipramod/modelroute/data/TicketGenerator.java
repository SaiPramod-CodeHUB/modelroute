package com.saipramod.modelroute.data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Generates realistic support tickets for 12 intents. Each intent has 5 template
 * "families" with two phrasings each. Families 0-3 form the historical distribution used for
 * train / validation / test (test tickets that duplicate a training ticket are dropped);
 * family 4 (casual slang) is held out entirely as a distribution-shift test. On top of the
 * templates: greetings, signatures, extra context, typos, lower-casing, tickets that mix two
 * issues, and a little label noise in training data.
 */
public final class TicketGenerator {

    public enum Split { TRAIN, VALIDATION, TEST, SHIFTED_TEST }

    public record Ticket(String text, String intent, int family) { }

    private static final String[] GREETINGS = {"Hi,", "Hello team,", "Hey there,", "Good morning,", "Hi support,", ""};
    private static final String[] SIGNOFFS = {"Thanks!", "Regards, Sam", "Sent from my iPhone", "Thank you", "Cheers", "- J", ""};
    private static final String[] CONTEXT = {"I've been a customer for 3 years.", "This is really frustrating.",
            "I already tried the help center.", "Not urgent but would appreciate a reply.", "This is the second time I'm writing.",
            "My colleague had the same problem."};

    private final Map<String, List<List<String>>> templates = new LinkedHashMap<>();
    private final Map<String, List<String>> slots = new LinkedHashMap<>();
    private final List<String> intents;

    public TicketGenerator() {
        try (InputStream in = TicketGenerator.class.getResourceAsStream("/data/templates.json")) {
            JsonNode root = new ObjectMapper().readTree(in);
            for (Iterator<Map.Entry<String, JsonNode>> it = root.get("intents").fields(); it.hasNext(); ) {
                var e = it.next();
                List<List<String>> fams = new ArrayList<>();
                e.getValue().forEach(f -> {
                    List<String> t = new ArrayList<>();
                    f.forEach(x -> t.add(x.asText()));
                    fams.add(t);
                });
                templates.put(e.getKey(), fams);
            }
            root.get("slots").fields().forEachRemaining(e -> {
                List<String> v = new ArrayList<>();
                e.getValue().forEach(x -> v.add(x.asText()));
                slots.put(e.getKey(), v);
            });
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        intents = List.copyOf(templates.keySet());
    }

    public List<String> intents() {
        return intents;
    }

    /** Generates tickets and drops any whose text already appears in {@code exclude} (no train/test overlap). */
    public List<Ticket> generateDisjoint(Split split, int n, long seed, java.util.Set<String> exclude) {
        List<Ticket> out = new ArrayList<>();
        long s = seed;
        while (out.size() < n) {
            for (Ticket t : generate(split, n, s++, 0.0)) {
                if (out.size() < n && exclude.add(t.text())) {
                    out.add(t);
                }
            }
        }
        return out;
    }

    public List<Ticket> generate(Split split, int n, long seed, double labelNoise) {
        Random r = new Random(seed);
        // Families 0-3 are the "historical" distribution (train / validation / test). Family 4, the
        // casual slang phrasings, is reserved for SHIFTED_TEST: wording no model saw in training.
        int[] families = split == Split.SHIFTED_TEST ? new int[] {4} : new int[] {0, 1, 2, 3};
        // Mild class imbalance, like a real queue: login/shipping/billing are the most common.
        double[] weights = new double[intents.size()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 1.0 + (i % 4 == 0 ? 0.8 : i % 3 == 0 ? 0.4 : 0);
        }
        List<Ticket> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            String intent = intents.get(weighted(r, weights));
            int fam = families[r.nextInt(families.length)];
            String body = fill(pick(r, templates.get(intent).get(fam)), r);
            if (r.nextDouble() < 0.15) {   // a second, secondary issue; label stays the primary one
                String other = intents.get(r.nextInt(intents.size()));
                body = body + " Also, " + lowerFirst(fill(pick(r, templates.get(other).get(fam)), r));
            }
            StringBuilder sb = new StringBuilder();
            String g = pick(r, GREETINGS);
            if (r.nextDouble() < 0.4 && !g.isEmpty()) {
                sb.append(g).append(' ');
            }
            if (r.nextDouble() < 0.2) {
                sb.append(pick(r, CONTEXT)).append(' ');
            }
            sb.append(body);
            if (r.nextDouble() < 0.35) {
                String s = pick(r, SIGNOFFS);
                if (!s.isEmpty()) {
                    sb.append(' ').append(s);
                }
            }
            String text = sb.toString();
            if (r.nextDouble() < 0.2) {
                text = text.toLowerCase();
            }
            text = typos(text, r, 0.04);
            String label = intent;
            if (labelNoise > 0 && r.nextDouble() < labelNoise) {
                label = intents.get(r.nextInt(intents.size()));
            }
            out.add(new Ticket(text, label, fam));
        }
        return out;
    }

    private String fill(String template, Random r) {
        String s = template;
        for (var e : slots.entrySet()) {
            while (s.contains("{" + e.getKey() + "}")) {
                s = s.replaceFirst("\\{" + e.getKey() + "}", java.util.regex.Matcher.quoteReplacement(pick(r, e.getValue())));
            }
        }
        return s;
    }

    private static String typos(String s, Random r, double rate) {
        char[] c = s.toCharArray();
        for (int i = 1; i < c.length - 1; i++) {
            if (Character.isLetter(c[i]) && Character.isLetter(c[i + 1]) && r.nextDouble() < rate) {
                char t = c[i];
                c[i] = c[i + 1];
                c[i + 1] = t;
                i++;
            }
        }
        return new String(c);
    }

    private static String lowerFirst(String s) {
        return s.isEmpty() ? s : Character.toLowerCase(s.charAt(0)) + s.substring(1);
    }

    private static int weighted(Random r, double[] w) {
        double total = 0;
        for (double x : w) {
            total += x;
        }
        double p = r.nextDouble() * total;
        for (int i = 0; i < w.length; i++) {
            p -= w[i];
            if (p <= 0) {
                return i;
            }
        }
        return w.length - 1;
    }

    private static <T> T pick(Random r, List<T> list) {
        return list.get(r.nextInt(list.size()));
    }

    private static String pick(Random r, String[] arr) {
        return arr[r.nextInt(arr.length)];
    }
}
