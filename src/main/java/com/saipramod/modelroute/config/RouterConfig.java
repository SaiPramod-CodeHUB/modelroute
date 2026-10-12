package com.saipramod.modelroute.config;

import com.saipramod.modelroute.model.Classifier;
import com.saipramod.modelroute.model.HttpClassifier;
import com.saipramod.modelroute.model.LlmClassifier;
import com.saipramod.modelroute.model.Pipeline;
import com.saipramod.modelroute.router.ConfidenceRouter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Startup: run the offline pipeline (generate data, train the large stand-in, distil and
 * calibrate the small model, pick the threshold). Then swap in real models if configured:
 * SMALL_MODEL_URL -> the LoRA model served by finetune/serve.py; LLM_PROVIDER=anthropic ->
 * Claude as the fallback model.
 */
@Configuration
public class RouterConfig {

    private static final Logger log = LoggerFactory.getLogger(RouterConfig.class);

    @Bean
    Pipeline.Trained trained() {
        Pipeline.Trained t = Pipeline.run(Pipeline.Config.defaults(), null);
        log.info("Trained in {} ms: threshold {} (validation fallback {}%)", t.trainingMillis(),
                t.threshold().threshold(), Math.round(t.threshold().fallbackRate() * 1000) / 10.0);
        return t;
    }

    @Bean
    ConfidenceRouter router(Pipeline.Trained t, ObjectProvider<ChatModel> chatModel,
                            @Value("${modelroute.small-model-url:}") String smallUrl) {
        Classifier small = smallUrl.isBlank() ? t.smallDistilled() : new HttpClassifier(smallUrl, t.labels());
        ChatModel m = chatModel.getIfAvailable();
        Classifier large = m == null ? t.large() : new LlmClassifier(ChatClient.create(m), t.labels());
        log.info("Router: small={} large={}", small.name(), large.name());
        return new ConfidenceRouter(small, large, t.threshold().threshold());
    }
}
