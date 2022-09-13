package neureka.autograd;

import neureka.Tsr;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 *  This interface is the declaration for
 *  lambda actions for both the {@link #act(ADTarget)} method of the {@link ADAgent} interface. <br><br>
 *  Implementations of this perform auto-differentiation forwards or backwards along the computation graph.
 *
 *
 * Note: Do not access the {@link GraphNode#getPayload()} of the {@link GraphNode}
 *       passed to implementation of this.
 *       The payload is weakly referenced, meaning that this method can return null!
 */
@FunctionalInterface
public interface ADAction
{
    /**
     *  The auto-differentiation forward or backward pass of an ADAgent
     *  propagate partial differentiations forward along the computation graph.
     *
     * @param target A wrapper for the {@link GraphNode} at which the differentiation ought to
     *               be performed and error which ought to be used for the forward or backward differentiation.
     * @return The result of a forward or backward mode auto differentiation.
     */
    Tsr<?> act( ADTarget<?> target );

    /**
     *  Finds captured {@link Tsr} instances in this current action
     *  using reflection (This is usually a partial derivative).
     *
     * @return The captured {@link Tsr} instances.
     */
    default Tsr<?>[] findCaptured() {
        List<Tsr<?>> captured = new ArrayList<>();
        for ( Class<?> c = this.getClass(); c != null; c = c.getSuperclass() ) {
            for ( java.lang.reflect.Field f : c.getDeclaredFields() ) {
                if ( f.getType().equals(Tsr.class) ) {
                    f.setAccessible(true);
                    try {
                        captured.add( (Tsr<?>) f.get(this) );
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return captured.toArray( new Tsr[0] );
    }

    default Optional<Tsr<?>> partialDerivative() {
        Tsr<?>[] captured = this.findCaptured();
        if ( captured.length > 0 )
            return Optional.of(captured[captured.length - 1]);

        return Optional.empty();
    }
}
