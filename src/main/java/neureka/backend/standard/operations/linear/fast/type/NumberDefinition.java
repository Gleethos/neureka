/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.type;

import java.util.Objects;

public interface NumberDefinition {

    static double doubleValue(final Comparable<?> number) {
        Objects.requireNonNull(number);
        if (number instanceof NumberDefinition) {
            return ((NumberDefinition) number).doubleValue();
        } else if (number instanceof Number) {
            return ((Number) number).doubleValue();
        } else {
            return Double.NaN;
        }
    }

    static float floatValue(final Comparable<?> number) {

        Objects.requireNonNull(number);

        if (number instanceof NumberDefinition) {
            return ((NumberDefinition) number).floatValue();
        } else if (number instanceof Number) {
            return ((Number) number).floatValue();
        } else {
            return Float.NaN;
        }
    }

    static int toInt(final double value) {
        return Math.round((float) value);
    }

    static long toLong(final double value) {
        return Math.round(value);
    }

    double doubleValue();

    default float floatValue() {
        return (float) this.doubleValue();
    }

}
