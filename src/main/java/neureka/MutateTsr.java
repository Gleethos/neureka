package neureka;

import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.common.utility.LogUtil;
import neureka.ndim.config.NDConfiguration;

import java.util.List;
import java.util.Map;

/**
 * Tensors should be considered immutable, however sometimes it
 * is important to mutate their state for performance reasons.
 * This interface exposes several methods for mutating the state of this tensor.
 * The usage of methods exposed by this API is generally discouraged
 * because the exposed state can easily lead to broken tensors and exceptions...<br>
 * <br>
 */
public interface MutateTsr<T> extends MutateNda<T>
{

    /** {@inheritDoc} */
    @Override Tsr<T> putAt( Map<?,Integer> key, Nda<T> value );

    @Override Tsr<T> putAt( int[] indices, T value );

    /** {@inheritDoc} */
    @Override default Tsr<T> set( int[] indices, T value ) { return putAt( indices, value ); }

    /** {@inheritDoc} */
    @Override default Tsr<T> set( int i0, int i1, T value ) { return putAt( new int[]{i0, i1}, value ); }

    /** {@inheritDoc} */
    @Override default Tsr<T> set( int i0, int i1, int i2, T value ) { return putAt( new int[]{i0, i1, i2}, value ); }

    /** {@inheritDoc} */
    @Override Tsr<T> putAt( int index, T value );

    /** {@inheritDoc} */
    @Override default Tsr<T> set( int index, T value ) { return putAt( index, value ); }

    /** {@inheritDoc} */
    @Override Tsr<T> putAt( List<?> key, Nda<T> value );

    /** {@inheritDoc} */
    @Override Tsr<T> putAt( List<?> indices, T value );

    /** {@inheritDoc} */
    @Override Tsr<T> setItemAt( int i, T o );

    /** {@inheritDoc} */
    @Override Tsr<T> setItems( Object value );

    /**
     *  This method takes the provided {@link Tsr} instance and adds its
     *  contents to the contents of the {@link Tsr} which is set as gradient of this very {@link Tsr}.
     *
     * @param error The error gradient which ought to be added to the gradient of this tensor.
     * @return This very tensor instance to enable method chaining.
     */
    Tsr<T> addToGradient( Tsr<T> error );

    /**
     * This method sets the NDConfiguration of this NDArray.
     * Therefore, it should not be used lightly as it can cause major internal inconsistencies.
     *
     * @param configuration The new NDConfiguration instance which ought to be set.
     * @return The final instance type of this class which enables method chaining.
     */
    Tsr<T> setNDConf(NDConfiguration configuration);

    /**
     * {@inheritDoc}
     */
    @Override
    <V> Tsr<V> toType( Class<V> typeClass );

    /**
     * Use this to do a runtime checked upcast of the type parameter of the tensor.
     * This is unsafe because it is in conflict with the {@link Tsr#itemType()}
     * method.
     *
     * @param superType The class of the super type of the tensor's value type.
     * @param <U>       The super type parameter of the value type of the tensor.
     * @return A tensor whose type parameter is upcast.
     */
    <U/*super T*/> Tsr<U> upcast(Class<U> superType);

    /**
     * This method allows you to modify the data-layout of this {@link AbstractNda}.
     * Warning! The method should not be used unless absolutely necessary.
     * This is because it can cause unpredictable side effects especially for certain
     * operations expecting a particular data layout (like for example matrix multiplication).
     * <br>
     *
     * @param layout The layout of the data array (row or column major).
     * @return The final instance type of this class which enables method chaining.
     */
    Tsr<T> toLayout(NDConfiguration.Layout layout);

    /**
     * This method is responsible for incrementing
     * the "_version" field variable which represents the version of the data of this tensor.
     * Meaning :
     * Every time the underlying data (_value) changes this version ought to increment alongside.
     * The method is called during the execution procedure.
     *
     * @param call The context object containing all relevant information that defines a call for tensor execution.
     * @return This very tensor instance. (factory pattern)
     */
    Tsr<T> incrementVersion(ExecutionCall<?> call);

    /**
     * Intermediate tensors are internal non-user tensors which may be eligible
     * for deletion when further consumed by a {@link Function}.
     * For the casual user of Neureka, this flag should always be false!
     *
     * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
     *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
     * @return The tensor to which this unsafe API belongs.
     */
    Tsr<T> setIsIntermediate(boolean isIntermediate);

    /**
     * Although tensors will be garbage collected when they are not strongly referenced,
     * there is also the option to manually free up the tensor and its associated data in a native environment.
     * This is especially useful when tensors are stored on a device like the {@link neureka.devices.opencl.OpenCLDevice}.
     * In that case calling this method will free the memory reserved for this tensor on the device.
     * This manual memory freeing through this method can be faster than waiting for
     * the garbage collector to kick in at a latr point in time... <br>
     * <br>
     *
     * @return The tensor wo which this unsafe API belongs to allow for method chaining.
     */
    Tsr<T> delete();

    /**
     * A tensor ought to have some way to selectively modify its underlying data array.
     * This method simply overrides an element within this data array sitting at position "i".
     *
     * @param i The index of the data array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    Tsr<T> setDataAt(int i, T o);

    Tsr<T> setData(Data<T> data);

    /**
     * Use this to access the underlying writable data of this tensor if
     * you want to modify it.
     * This method will ensure that you receive an instance of whatever array type you provide
     * or throw descriptive exceptions to make sure that any unwanted behaviour does not
     * spread further in the backend.
     *
     * @param arrayTypeClass The expected array type underlying the tensor.
     * @param <A>            The type parameter of the provided type class.
     * @return The underlying data array of this tensor.
     */
    default <A> A getDataForWriting(Class<A> arrayTypeClass) {
        LogUtil.nullArgCheck(arrayTypeClass, "arrayTypeClass", Class.class, "Array type must not be null!");
        if (!arrayTypeClass.isArray())
            throw new IllegalArgumentException("Provided type is not an array type.");
        Object data = MutateTsr.this.getData().getRef();
        if (data == null)
            throw new IllegalStateException("Could not find writable tensor data for this tensor (Maybe this tensor is stored on a device?).");

        if (!arrayTypeClass.isAssignableFrom(data.getClass()))
            throw new IllegalStateException("The data of this tensor does not match the expect type! Expected '" + arrayTypeClass + "' but got '" + data.getClass() + "'.");

        return (A) data;
    }

    /**
     * <b>This method detaches this tensor from its underlying computation-graph
     * or simply does nothing if no graph is present.</b> <br>
     * Nodes within a computation graph are instances of the "{@link GraphNode}" class which are also
     * simple components of the tensors they represent in the graph. <br>
     * Therefore, "detaching" this tensor from the graph simply means removing its {@link GraphNode} component.
     *
     * @return This very instance in order to allows for a more streamline usage of this method.
     */
    Tsr<T> detach();

    /**
     * @param other The tensor whose elements ought to be multiplied and assigned to elements in this tensor.
     * @return This instance where each value element was multiplied by the corresponding element in the provided tensor.
     */
    Tsr<T> timesAssign(Tsr<T> other);

    /**
     * @param other The value which ought to be multiplied and assigned to each element in this tensor.
     * @return This instance where each value element was multiplied by the provided element.
     */
    Tsr<T> timesAssign(T other);

    Tsr<T> divAssign(Tsr<T> other);

    Tsr<T> modAssign(Tsr<T> other);

    /**
     * Performs an addition of the passed tensor to this tensor.
     * The result of the addition will be stored in this tensor (inline operation).
     *
     * @param other The tensor which ought to be added to this tensor.
     * @return This tensor.
     */
    Tsr<T> plusAssign(Tsr<T> other);

    Tsr<T> minusAssign(Tsr<T> other);

    /**
     * @param other The scalar value which should be subtracted from the values of this tensor.
     * @return This tensor after the minus-assign inline operation was applied.
     */
    Tsr<T> minusAssign(T other);

    @Override Tsr<T> assign( T other );

    @Override Tsr<T> assign( Nda<T> other );

    /** {@inheritDoc} */
    @Override Tsr<T> label( String label );

    /**
     * This method receives a label for this tensor and a
     * nested {@link String} array which ought to contain a
     * label for the index of this tensor.
     * The index for a single element of this tensor would be an array
     * of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows:    <br>
     * <br>
     * dim 0 : ["A", "B"]                                      <br>
     * dim 1 : ["1", "2", "3"]                                 <br>
     * <br>
     *
     * @param labels     A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<T> labelAxes( String[]... labels );

    /**
     * This method receives a nested {@link String} list which
     * ought to contain a label for the index of this tensor.
     * The index for a single element of this tensor would be an array
     * of numbers as long as the rank where every number is
     * in the range of the corresponding shape dimension...
     * Labeling an index means that for every dimension there
     * must be a label for elements in this range array! <br>
     * For example the shape (2,3) could be labeled as follows: <br>
     * <br>
     * dim 0 : ["A", "B"]                                   <br>
     * dim 1 : ["1", "2", "3"]                              <br>
     * <br>
     *
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<T> labelAxes( List<List<Object>> labels );

    /**
     * This method provides the ability to
     * label not only the indices of the shape of this tensor, but also
     * the dimension of the shape.
     * The first and only argument of the method expects a map instance
     * where keys are the objects which ought to act as dimension labels
     * and the values are lists of labels for the indices of said dimensions.
     * For example the shape (2,3) could be labeled as follows:            <br>
     * [                                                                   <br>
     * "dim 0" : ["A", "B"],                                           <br>
     * "dim 1" : ["1", "2", "3"]                                       <br>
     * ]                                                                   <br>
     * <br>
     *
     * @param labels A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This tensor (method chaining).
     */
    Tsr<T> labelAxes( Map<Object, List<Object>> labels );

    /**
     *  Virtualizing is the opposite to actualizing a tensor.
     *  A tensor is virtual if the size of the underlying data is not actually equal to
     *  the number of elements which the tensor claims to store, aka its size.
     *  This is for example the case when initializing a tensor filled with a single
     *  value continuously. In that case the tensor will flag itself as virtual and only allocate the
     *  underlying data array to hold a single item even though the tensor might actually hold
     *  many more items.
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *  <br><br>
     *  WARNING! Virtualizing is the process of compacting the underlying data array
     *  down to an array holding a single value item.
     *  This only makes sense for homogeneously populated tensors.
     *  Passing {@code false} to this method will "actualize" a "virtual" tensor.
     *  Meaning the underlying data array will at least become as large as the size of the tensor
     *  as is defined by {@link Tsr#size()}.
     *
     * @param isVirtual The truth value determining if this tensor should be "virtual" or "actual".
     * @return This concrete instance, to allow for method chaining.
     */
    Tsr<T> setIsVirtual( boolean isVirtual );

}
