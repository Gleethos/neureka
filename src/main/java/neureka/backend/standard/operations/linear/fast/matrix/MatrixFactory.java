/*<#LICENSE#>*/
package neureka.backend.standard.operations.linear.fast.matrix;

import neureka.backend.standard.operations.linear.fast.ProgrammingError;
import neureka.backend.standard.operations.linear.fast.function.FunctionSet;
import neureka.backend.standard.operations.linear.fast.matrix.store.MatrixCore;
import neureka.backend.standard.operations.linear.fast.structure.Access2D;
import neureka.backend.standard.operations.linear.fast.structure.Factory2D;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

/**
 * MatrixFactory creates instances of classes that implement the {@linkplain BasicMatrix}
 * interface and have a constructor that takes a MatrixStore as input.
 *
 */
public abstract class MatrixFactory<
                            N extends Comparable<N>,
                            M extends BasicMatrix<N, M>
                        >
                        implements Factory2D.Dense<M>
{

    private static Constructor<? extends BasicMatrix<?, ?>> getConstructor(final Class<? extends BasicMatrix<?, ?>> aTemplate) {
        try {
            final Constructor<? extends BasicMatrix<?, ?>> retVal = aTemplate.getDeclaredConstructor(MatrixCore.class);
            retVal.setAccessible(true);
            return retVal;
        } catch (final SecurityException | NoSuchMethodException exception) {
            return null;
        }
    }

    private final Constructor<M> _constructor;
    private final MatrixCore.Factory<N, ?> myPhysicalFactory;

    MatrixFactory(final Class<M> template, final MatrixCore.Factory<N, ?> factory) {

        super();

        myPhysicalFactory = factory;
        _constructor = (Constructor<M>) MatrixFactory.getConstructor(template);
    }

    @Override
    public M copy(final Access2D<?> source) {
        return this.instantiate(myPhysicalFactory.copy(source));
    }

    @Override
    public FunctionSet<N> forFunctions() {
        return myPhysicalFactory.forFunctions();
    }

    @Override
    public M makeFilled(final long rows, final long columns, final Object data) {
        return this.instantiate(myPhysicalFactory.makeFilled(rows, columns, data));
    }

    /**
     * This method is for internal use only - YOU should NOT use it!
     */
    M instantiate(final MatrixCore<N> store) {
        try {
            return _constructor.newInstance(store);
        } catch (final IllegalArgumentException | InstantiationException | IllegalAccessException | InvocationTargetException anException) {
            throw new ProgrammingError(anException);
        }
    }

}
