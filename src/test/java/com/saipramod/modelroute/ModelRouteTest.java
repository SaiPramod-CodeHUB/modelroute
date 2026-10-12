package com.saipramod.modelroute;

import com.saipramod.modelroute.data.TicketGenerator;
import com.saipramod.modelroute.data.TicketGenerator.Split;
import com.saipramod.modelroute.data.TicketGenerator.Ticket;
import com.saipramod.modelroute.eval.EvalRunner;
import com.saipramod.modelroute.model.Classifier;
import com.saipramod.modelroute.model.FeatureHasher;
import com.saipramod.modelroute.model.LinearClassifier;
import com.saipramod.modelroute.model.Pipeline;
import com.saipramod.modelroute.router.ConfidenceRouter;
import com.saipramod.modelroute.router.CostModel;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

class ModelRouteTest {

    static final Pipeline.Config SMALL = new Pipeline.Config(600, 1500, 400, 600, 7, 0.99);

    @Test
    void splitsShareNoTicketTextAndShiftedSetUsesUnseenPhrasing() {
        TicketGenerator g = new TicketGenerator();
        List<Ticket> train = g.generate(Split.TRAIN, 2000, 1, 0);
        Set<String> seen = new HashSet<>();
        train.forEach(t -> seen.add(t.text()));
        List<Ticket> test = g.generateDisjoint(Split.TEST, 500, 2, new HashSet<>(seen));
        assertThat(test).noneMatch(t -> seen.contains(t.text()));
        assertThat(g.generate(Split.SHIFTED_TEST, 200, 3, 0)).allMatch(t -> t.family() == 4);
        assertThat(train).allMatch(t -> t.family() < 4);
    }

    @Test
    void linearModelLearnsAndCalibrates() {
        TicketGenerator g = new TicketGenerator();
        List<Ticket> train = g.generate(Split.TRAIN, 1500, 1, 0);
        List<Ticket> val = g.generate(Split.VALIDATION, 300, 2, 0);
        LinearClassifier c = new LinearClassifier("t", new FeatureHasher(1 << 14, 2, 0, 0), g.intents()).fitGold(train, 4, 1);
        double t = c.calibrate(val);
        assertThat(t).isBetween(0.25, 6.0);
        long correct = val.stream().filter(x -> c.predict(x.text()).label().equals(x.intent())).count();
        assertThat((double) correct / val.size()).isGreaterThan(0.8);
        Classifier.Prediction p = c.predict("I was charged twice for my order");
        assertThat(p.label()).isEqualTo("DOUBLE_CHARGE");
        assertThat(java.util.Arrays.stream(p.probs()).sum()).isCloseTo(1.0, org.assertj.core.data.Offset.offset(1e-6));
    }

    @Test
    void routerFallsBackOnlyBelowThreshold() {
        Classifier sure = fixed("A", 0.95);
        Classifier unsure = fixed("A", 0.40);
        Classifier large = fixed("B", 0.99);
        ConfidenceRouter r = new ConfidenceRouter(sure, large, 0.6);
        assertThat(r.route("x").fellBack()).isFalse();
        ConfidenceRouter r2 = new ConfidenceRouter(unsure, large, 0.6);
        assertThat(r2.route("x").prediction().label()).isEqualTo("B");
        assertThat(r2.fallbackRate()).isEqualTo(1.0);
    }

    @Test
    void costModelChargesTokensForLargeAndComputeTimeForSmall() {
        CostModel c = CostModel.defaults();
        assertThat(c.largeCost("x".repeat(400))).isGreaterThan(c.smallCost(1_000));
    }

    @Test
    void evalGate() {
        EvalRunner.Report r = EvalRunner.run(SMALL, null);
        EvalRunner.Row large = r.rows().get(0);
        EvalRunner.Row distilled = r.rows().get(2);
        EvalRunner.Row router = r.rows().get(3);
        assertThat(distilled.accuracy()).isGreaterThan(0.80);
        assertThat(router.qualityRetained()).isGreaterThanOrEqualTo(0.97);
        assertThat(router.costPer1k()).isLessThan(large.costPer1k() / 2);
        // under distribution shift the router must fall back more and do at least as well as the small model alone
        assertThat(r.shiftedRows().get(3).fallbackRate()).isGreaterThan(router.fallbackRate());
        assertThat(r.shiftedRows().get(3).accuracy()).isGreaterThanOrEqualTo(r.shiftedRows().get(2).accuracy());
    }

    private static Classifier fixed(String label, double conf) {
        return new Classifier() {
            public Prediction predict(String text) {
                return new Prediction(label, conf, new double[] {conf, 1 - conf}, 1, label);
            }
            public List<String> labels() { return List.of("A", "B"); }
            public String name() { return "fixed-" + label; }
        };
    }
}
