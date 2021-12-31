/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.structure;

public interface Access1D<N extends Comparable<N>> extends Size {

    static String toString(final Access1D<?> access) {
        int size = access.size();
        switch (size) {
        case 0: return "{ }";
        case 1: return "{ " + access.get(0) + " }";
        default:
            StringBuilder builder = new StringBuilder();
            builder.append("{ ");
            builder.append(access.get(0));
            for (int i = 1; i < size; i++) {
                builder.append(", ");
                builder.append(access.get(i));
            }
            builder.append(" }");
            return builder.toString();
        }
    }

    double doubleValue(long index);

    default float floatValue(final long index) {
        return (float) this.doubleValue(index);
    }

    N get(long index);

}
