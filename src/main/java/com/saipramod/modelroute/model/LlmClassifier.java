package com.saipramod.modelroute.model;

import org.springframework.ai.chat.client.ChatClient;

import java.util.List;

/**
 * The frontier model as a classifier, via Spring AI structured output. This is the
 * baseline to beat and the router's fallback; it is also the "teacher" that labels the
 * unlabeled pool for distillation when LLM_PROVIDER is set.
 */
public final class LlmClassifier implements Classifier {

    public record Answer(String intent, double confidence) { }

    private final ChatClient chat;
    private final List<String> labels;
    private final String system;

    public LlmClassifier(ChatClient chat, List<String> labels) {
        this.chat = chat;
        this.labels = labels;
        this.system = "Classify the customer support ticket into exactly one intent: " + String.join(", ", labels)
                + ". If the ticket mentions several issues, choose the one raised first. "
                + "Return the intent and your confidence between 0 and 1.";
    }

    @Override
    public Prediction predict(String text) {
        long t0 = System.nanoTime();
        Answer a;
        try {
            a = chat.prompt().system(system).user(text).call().entity(Answer.class);
        } catch (RuntimeException e) {
            a = null;
        }
        double[] p = new double[labels.size()];
        int idx = a == null ? -1 : labels.indexOf(a.intent());
        if (idx < 0) {
            java.util.Arrays.fill(p, 1.0 / labels.size());
            idx = 0;
        } else {
            double conf = Math.max(0, Math.min(1, a.confidence()));
            java.util.Arrays.fill(p, (1 - conf) / (labels.size() - 1));
            p[idx] = conf;
        }
        return new Prediction(labels.get(idx), p[idx], p, (System.nanoTime() - t0) / 1000, name());
    }

    @Override
    public List<String> labels() {
        return labels;
    }

    @Override
    public String name() {
        return "frontier LLM (Spring AI)";
    }
}
