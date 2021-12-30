/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.function;

import neureka.backend.standard.operations.linear.fast.type.NumberDefinition;

import java.util.function.BiFunction;

public interface ParameterFunction<N extends Comparable<N>> extends BiFunction<N, Integer, N> {

    final class FixedParameter<N extends Comparable<N>> implements UnaryFunction<N> {

        private final ParameterFunction<N> myFunction;
        private final int myParameter;

        @SuppressWarnings("unused")
        private FixedParameter() {
            this(null, 0);
        }

        FixedParameter(final ParameterFunction<N> function, final int param) {

            super();

            myFunction = function;
            myParameter = param;
        }

        public ParameterFunction<N> getFunction() {
            return myFunction;
        }

        public int getParameter() {
            return myParameter;
        }

        public double invoke(final double arg) {
            return myFunction.invoke(arg, myParameter);
        }

        public float invoke(final float arg) {
            return myFunction.invoke(arg, myParameter);
        }

        public N invoke(final N arg) {
            return myFunction.invoke(arg, myParameter);
        }

    }

    default N apply(final N arg, final Integer param) {
        return this.invoke(arg, param.intValue());
    }

    default byte invoke(final byte arg, final int param) {
        return (byte) this.invoke((double) arg, param);
    }

    double invoke(double arg, int param);

    default float invoke(final float arg, final int param) {
        return (float) this.invoke((double) arg, param);
    }

    default long invoke(final long arg, final int param) {
        return NumberDefinition.toLong(this.invoke((double) arg, param));
    }

    N invoke(N arg, int param);

    default short invoke(final short arg, final int param) {
        return (short) this.invoke((double) arg, param);
    }

}
