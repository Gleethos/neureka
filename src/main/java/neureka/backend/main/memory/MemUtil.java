package neureka.backend.main.memory;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *  Utility methods for deleting tensors or preventing thereof.
 *  In essence, it exposes convenience methods for setting and resetting
 *  the {@link Tsr#isIntermediate} flag or supplied tensors...
 *  This is an internal library class which should not be used
 *  anywhere but in Neurekas backend.
 *  <b>Do not use this anywhere else!</b>
 */
public class MemUtil {

    private MemUtil() {}

    /**
     *  This method will try to delete the provided array of tensors
     *  if the tensors are not important computation
     *  graph components (like derivatives for example).
     *
     * @param tensors The tensors which should be deleted if possible.
     */
    public static void autoDelete( Tsr<?>... tensors ) {
        /*
             When we are purely in the JVM world, then the garbage
             collector will take care of freeing our memory,
             and we don't really have a saying in when something gets collected...
             However, this is different for native memory (for example the GPU memory)!
             In that case we can manually free up the data array of a tensor.
             The code below will delete intermediate tensors which are expected
             to be no longer used.
        */
        if ( Neureka.get().settings().debug().isDeletingIntermediateTensors() ) {
            for ( Tsr<?> t : tensors ) {
                // Tensors flagged as 'intermediate' will automatically be deleted!
                if ( !t.isDeleted() && t.isIntermediate() ) {
                    if (
                        t.getGraphNode()
                        .map(n->!n.isUsedAsDerivative())
                        .orElse(true) // No graph, we can delete it!
                    )
                        t.mut().delete();
                }
            }
        }
    }

    /**
     *  This method makes sure that the provided tensors do not get deleted
     *  by setting the {@link Tsr#isIntermediate} flag to off
     *  during the execution of the provided {@link Supplier} lambda!
     *  In said lambda the supplied thing will ultimately be returned by
     *  this method...
     *  All provided tensors will have the {@link Tsr#isIntermediate} flag
     *  set to their original state after execution.
     *
     * @param tensors An array of tensors which should not be deleted during the execution of the supplied lambda.
     * @param during A lambda producing a result during which the provided tensors should not be deleted.
     * @param <T> The type of the result produced by the provided lambda.
     * @return The result produced by the provided lambda.
     */
    public static <T> T keep( Tsr<?>[] tensors, Supplier<T> during ) {
        List<Tsr<?>> doNotDelete = Arrays.stream(tensors).filter(Tsr::isIntermediate).collect(Collectors.toList());
        doNotDelete.forEach( t -> t.mut().setIsIntermediate( false ) );
        T result = during.get();
        // After having calculated the result we allow deletion of the provided tensors again:
        doNotDelete.forEach( t -> t.mut().setIsIntermediate( true ) );
        return result;
    }

    /**
     *  This method makes sure that the provided tensors do not get deleted
     *  by setting the {@link Tsr#isIntermediate} flag to off
     *  during the execution of the provided {@link Supplier} lambda!
     *  In said lambda the supplied thing will ultimately be returned by
     *  this method...
     *  Both of the provided tensors will have the {@link Tsr#isIntermediate} flag
     *  set to their original state after execution.
     *
     * @param a The first tensor which should not be deleted during the execution of the provided lambda.
     * @param b The second tensor which should not be deleted during the execution of the provided lambda.
     * @param during A lambda producing a result during whose execution the first to arguments should not be deleted.
     * @param <T> The type of the result produced by the provided lambda.
     * @return The result produced by the provided lambda.
     */
    public static <T> T keep( Tsr<?> a, Tsr<?> b, Supplier<T> during ) {
        List<Tsr<?>> doNotDelete = Stream.of(a, b).filter(Tsr::isIntermediate).collect(Collectors.toList());
        doNotDelete.forEach( t -> t.mut().setIsIntermediate( false ) );
        T result = during.get();
        // After having calculated the result we allow deletion of the provided tensors again:
        doNotDelete.forEach( t -> t.mut().setIsIntermediate( true ) );
        return result;
    }

}
