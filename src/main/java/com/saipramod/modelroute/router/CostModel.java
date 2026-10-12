package com.saipramod.modelroute.router;

/**
 * Cost assumptions, all configurable. The large model is billed per token like a hosted
 * frontier API; the small model is billed by the compute time it occupies on a
 * self-hosted instance. Defaults are illustrative; plug in your provider's real prices.
 *
 * @param largeInputPerMTok   USD per million input tokens for the large model
 * @param largeOutputPerMTok  USD per million output tokens for the large model
 * @param promptOverheadTokens instruction + label-list tokens sent with every request
 * @param outputTokens        tokens in the large model's answer
 * @param smallInstancePerHour USD per hour for the small model's serving instance
 * @param smallUtilisation    fraction of instance time doing useful work
 */
public record CostModel(double largeInputPerMTok, double largeOutputPerMTok, int promptOverheadTokens, int outputTokens,
                        double smallInstancePerHour, double smallUtilisation) {

    public static CostModel defaults() {
        return new CostModel(3.0, 15.0, 220, 12, 0.10, 0.5);
    }

    public double largeCost(String ticket) {
        int in = promptOverheadTokens + Math.max(1, ticket.length() / 4);
        return (in * largeInputPerMTok + outputTokens * largeOutputPerMTok) / 1_000_000.0;
    }

    public double smallCost(long computeMicros) {
        double seconds = computeMicros / 1_000_000.0 / smallUtilisation;
        return smallInstancePerHour / 3600.0 * seconds;
    }
}
