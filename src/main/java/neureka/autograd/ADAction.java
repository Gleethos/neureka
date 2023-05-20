package neureka.autograd;

import neureka.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 *  This interface is the declaration for
 *  lambda actions for both the {@link #act(ADTarget)} method of the {@link ADAction} interface. <br><br>
 *  Implementations of this perform auto-differentiation forwards or backwards along the computation graph.
 *  These differentiation actions are performed through the "{@link ADAction#act(ADTarget)}"
 *  method which are being called
 *  by instances of the {@link GraphNode} class during propagation.
 *  An {@link ADAction} may also wrap and expose a partial derivative
 *  which may or may not be present for certain operations.
 *  Said derivative must be tracked and flagged as derivative by a {@link GraphNode}
 *  to make sure that it will not be deleted after a forward pass.
 *  <p>
 * Note: Do not access the {@link GraphNode#getPayload()} of the {@link GraphNode}
 *       passed to implementation of this.
 *       The payload is weakly referenced, meaning that this method can return null!
 */
@FunctionalInterface
public interface ADAction
{
    static ADAction of( ADAction action ) { return new DefaultADAction( action, null ); }

    static ADAction of(Tensor<?> derivative, ADAction action ) { return new DefaultADAction( action, derivative ); }

    /**
     *  The auto-differentiation forward or backward pass of an ADAction
     *  propagate partial differentiations forward along the computation graph.
     *
     * @param target A wrapper for the {@link GraphNode} at which the differentiation ought to
     *               be performed and error which ought to be used for the forward or backward differentiation.
     * @return The result of a forward or backward mode auto differentiation.
     */
    Tensor<?> act(ADTarget<?> target );

    /**
     *  Finds captured {@link Tensor} instances in this current action
     *  using reflection (This is usually a partial derivative).
     *
     * @return The captured {@link Tensor} instances.
     */
    default Tensor<?>[] findCaptured() {
        List<Tensor<?>> captured = new ArrayList<>();
        for ( Class<?> c = this.getClass(); c != null; c = c.getSuperclass() ) {
            for ( java.lang.reflect.Field f : c.getDeclaredFields() ) {
                if ( f.getType().equals(Tensor.class) ) {
                    f.setAccessible(true);
                    try {
                        captured.add( (Tensor<?>) f.get(this) );
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        return captured.toArray( new Tensor[0] );
    }

    default Optional<Tensor<?>> partialDerivative() {
        Tensor<?>[] captured = this.findCaptured();
        if ( captured.length > 0 )
            return Optional.of(captured[captured.length - 1]);

        return Optional.empty();
    }
}
