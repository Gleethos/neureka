/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import java.util.function.Consumer;
import java.util.function.DoubleConsumer;

public interface VoidFunction<N extends Comparable<N>> extends Consumer<N>, DoubleConsumer {

    default void accept(final double arg) {
        this.invoke(arg);
    }

    default void accept(final N arg) {
        this.invoke(arg);
    }

    default void invoke(final byte arg) {
        this.invoke((double) arg);
    }

    void invoke(double arg);

    default void invoke(final float arg) {
        this.invoke((double) arg);
    }

    default void invoke(final int arg) {
        this.invoke((double) arg);
    }

    default void invoke(final long arg) {
        this.invoke((double) arg);
    }

    void invoke(N arg);

    default void invoke(final short arg) {
        this.invoke((double) arg);
    }

}
