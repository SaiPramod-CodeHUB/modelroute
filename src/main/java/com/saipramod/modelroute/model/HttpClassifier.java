package com.saipramod.modelroute.model;

import org.springframework.web.client.RestClient;

import java.util.List;
import java.util.Map;

/**
 * Client for the LoRA fine-tuned model served by {@code finetune/serve.py}
 * (POST /classify {"text": ...} -> {"label", "probs"}). Lets the Java router use a real
 * fine-tuned open-weight model as the small model without changing anything else.
 */
public final class HttpClassifier implements Classifier {

    record Response(String label, List<Double> probs) { }

    private final RestClient http;
    private final List<String> labels;

    public HttpClassifier(String baseUrl, List<String> labels) {
        this.http = RestClient.builder().baseUrl(baseUrl).build();
        this.labels = labels;
    }

    @Override
    public Prediction predict(String text) {
        long t0 = System.nanoTime();
        Response r = http.post().uri("/classify").body(Map.of("text", text)).retrieve().body(Response.class);
        double[] p = new double[labels.size()];
        if (r != null && r.probs() != null && r.probs().size() == labels.size()) {
            for (int i = 0; i < p.length; i++) {
                p[i] = r.probs().get(i);
            }
        } else if (r != null && labels.contains(r.label())) {
            p[labels.indexOf(r.label())] = 1;
        }
        int k = Classifier.argmax(p);
        return new Prediction(labels.get(k), p[k], p, (System.nanoTime() - t0) / 1000, name());
    }

    @Override
    public List<String> labels() {
        return labels;
    }

    @Override
    public String name() {
        return "lora-finetuned (HTTP)";
    }
}
