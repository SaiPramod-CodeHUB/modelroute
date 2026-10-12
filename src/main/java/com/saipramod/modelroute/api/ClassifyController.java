package com.saipramod.modelroute.api;

import com.saipramod.modelroute.model.Pipeline;
import com.saipramod.modelroute.router.ConfidenceRouter;
import jakarta.validation.Valid;
import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ClassifyController {

    public record ClassifyRequest(@NotBlank @Size(max = 5000) String text) { }

    public record ClassifyResponse(String intent, double confidence, String servedBy, boolean fellBack,
                                  String smallModelGuess, double smallModelConfidence, long latencyMicros) { }

    public record Threshold(@DecimalMin("0.0") @DecimalMax("1.0") double value) { }

    private final ConfidenceRouter router;
    private final Pipeline.Trained trained;

    public ClassifyController(ConfidenceRouter router, Pipeline.Trained trained) {
        this.router = router;
        this.trained = trained;
    }

    @PostMapping("/classify")
    public ClassifyResponse classify(@Valid @RequestBody ClassifyRequest req) {
        ConfidenceRouter.Routed r = router.route(req.text());
        return new ClassifyResponse(r.prediction().label(), round(r.prediction().confidence()), r.prediction().model(),
                r.fellBack(), r.small().label(), round(r.small().confidence()), r.prediction().latencyMicros());
    }

    @PostMapping("/threshold")
    public Map<String, Object> threshold(@Valid @RequestBody Threshold t) {
        router.setThreshold(t.value());
        return stats();
    }

    @GetMapping("/stats")
    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("router", router.name());
        m.put("threshold", router.threshold());
        m.put("requests", router.requests());
        m.put("fallbackRate", round(router.fallbackRate()));
        m.put("validationChoice", trained.threshold());
        m.put("smallModel", trained.smallDistilled().features() + ", " + trained.smallDistilled().parameters() + " params");
        m.put("labels", trained.labels());
        return m;
    }

    private static double round(double v) {
        return Math.round(v * 1000) / 1000.0;
    }
}
