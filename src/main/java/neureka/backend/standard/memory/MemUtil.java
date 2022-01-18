package neureka.backend.standard.memory;

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
 */
public class MemUtil {

    private MemUtil() {}

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
                    GraphNode<?> node = t.getGraphNode();
                    if ( node == null || !node.isUsedAsDerivative() )
                        t.delete();
                }
            }
        }
    }

    /**
     *  This method makes sure that the provided tensors do not get deleted!
     */
    public static <T> T keep(Tsr<?>[] tensors, Supplier<T> during ) {
        List<Tsr<?>> doNotDelete = Arrays.stream(tensors).filter(Tsr::isIntermediate).collect(Collectors.toList());
        doNotDelete.forEach( t -> t.getMutate().setIsIntermediate( false ) );
        T result = during.get();
        // After having calculated the result we allow deletion of the provided tensors again:
        doNotDelete.forEach( t -> t.getMutate().setIsIntermediate( true ) );
        return result;
    }

    /**
     *  This method makes sure that the provided tensors do not get deleted!
     */
    public static <T> T keep( Tsr<?> a, Tsr<?> b, Supplier<T> during ) {
        List<Tsr<?>> doNotDelete = Stream.of(a, b).filter(Tsr::isIntermediate).collect(Collectors.toList());
        doNotDelete.forEach( t -> t.getMutate().setIsIntermediate( false ) );
        T result = during.get();
        // After having calculated the result we allow deletion of the provided tensors again:
        doNotDelete.forEach( t -> t.getMutate().setIsIntermediate( true ) );
        return result;
    }

}
