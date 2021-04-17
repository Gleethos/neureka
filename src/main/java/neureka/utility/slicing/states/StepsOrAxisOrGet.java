package neureka.utility.slicing.states;


/**
 *  This interface extends the {@link AxisOrGet} interface which provides the option to either continue
 *  slicing another axis or simply trigger the creation and return of a slice instance based on the
 *  already provided slice configuration.
 *  The method signature introduced in this interface provides the possibility to set a step size
 *  for the previously defined range ({@link FromOrAt#from(int)} and {@link To#to(int)}).
 *  This step size will be used to create strides within the sliced axis.
 *
 * @param <V> The type parameter for items of the {@link neureka.Tsr} which ought to be sliced.
 */
public interface StepsOrAxisOrGet<V> extends AxisOrGet<V>
{
    AxisOrGet<V> step(int size );
}
