package com.saipramod.modelroute.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.saipramod.modelroute.data.TicketGenerator.Split;
import com.saipramod.modelroute.data.TicketGenerator.Ticket;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Writes the dataset as JSONL for the Python LoRA fine-tuning step, with the same splits
 * and the same no-overlap guarantee the Java eval uses.
 *
 * <pre>mvn -q compile exec:java -Dexec.mainClass=com.saipramod.modelroute.data.DatasetExporter -Dexec.args="data 7500"</pre>
 */
public final class DatasetExporter {

    private DatasetExporter() { }

    public static void main(String[] args) throws IOException {
        Path dir = Path.of(args.length > 0 ? args[0] : "data");
        int trainN = args.length > 1 ? Integer.parseInt(args[1]) : 7500;
        Files.createDirectories(dir);
        TicketGenerator gen = new TicketGenerator();
        List<Ticket> train = gen.generate(Split.TRAIN, trainN, 42, 0.02);
        Set<String> seen = new HashSet<>();
        train.forEach(t -> seen.add(t.text()));
        write(dir.resolve("train.jsonl"), train);
        write(dir.resolve("validation.jsonl"), gen.generateDisjoint(Split.VALIDATION, 1000, 44, seen));
        write(dir.resolve("test.jsonl"), gen.generateDisjoint(Split.TEST, 2000, 45, seen));
        write(dir.resolve("shifted_test.jsonl"), gen.generateDisjoint(Split.SHIFTED_TEST, 1000, 46, seen));
        Files.writeString(dir.resolve("labels.json"), new ObjectMapper().writeValueAsString(gen.intents()));
        System.out.println("Wrote dataset to " + dir.toAbsolutePath());
    }

    private static void write(Path p, List<Ticket> tickets) throws IOException {
        ObjectMapper om = new ObjectMapper();
        List<String> lines = new ArrayList<>();
        for (Ticket t : tickets) {
            lines.add(om.writeValueAsString(Map.of("text", t.text(), "label", t.intent())));
        }
        Files.write(p, lines);
    }
}
