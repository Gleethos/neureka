package neureka.ndim;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.calculus.Functions;
import neureka.common.utility.DataConverter;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.dtype.custom.UI16;
import neureka.dtype.custom.UI32;
import neureka.dtype.custom.UI8;
import neureka.fluent.slicing.SliceBuilder;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.ndim.config.NDConfiguration;
import neureka.view.TsrAsString;
import neureka.view.TsrStringSettings;

import java.awt.image.BufferedImage;
import java.util.*;
import java.util.function.Consumer;

/**
 *  This interface is part of the {@link Tsr} API, and it defines
 *  how data can be read from and written to a tensor.
 *  In essence, this interface exists to expand
 *  the tensor API through default methods without littering the
 *  already large {@link Tsr} and {@link AbstractTensor} classes.
 *
 * @param <V> The value type parameter of the items stored by this tensor.
 */
public interface TensorAPI<V> extends NDimensional, Iterable<V>
{
    /**
     *  This will check if the {@link Unsafe#delete()} method was previously called on this tensor.
     *  This means that any references inside the tensor will be null
     *  as well as that the tensor data was freed on every device,
     *  meaning that what was previously referenced was most likely garbage collected...
     *
     * @return The truth value determining if the {@link Unsafe#delete()} method has been called oin this instance.
     */
    boolean isDeleted();

    /**
     *  A Virtual tensor is a tensor whose underlying data array is of size 1, holding only a single value. <br>
     *  This only makes sense for homogeneously populated tensors.
     *  An example of such a tensor would be: <br>
     *  {@code Tsr.ofInts().withShape(x,y).all(n)}                           <br><br>
     *  The reasons for this feature is that it greatly improves performance in certain cases.
     *  In essence this feature is a form of lazy loading.
     *
     *  Use {@link #setIsVirtual(boolean)} to "actualize" a "virtual" tensor, and vise versa.
     *
     * @return The truth value determining if this tensor is "virtual" or "actual".
     */
    boolean isVirtual();

    /**
     *  A tensor is empty if there is neither data referenced within the tensor directly
     *  nor within any given device to which the tensor might belong.
     *
     * @return The truth value determining if this tensor has data.
     */
    boolean isEmpty();

    /**
     *  A tensor is "undefined" if it has either no {@link NDConfiguration} implementation instance
     *  or this instance does not have a shape set for this {@link Tsr} which is needed for
     *  a tensor to also have a rank and dimensionality...
     *
     * @return The truth value determining if this tensor has an {@link NDConfiguration} stored internally.
     */
    boolean isUndefined();

    /**
     *  If this tensor is a slice of a parent tensor then this method will yield true.
     *  Slices can be created by calling the variations of the "{@link Tsr#getAt}" method.
     *
     * @return The truth value determining if this tensor is a slice of another tensor.
     */
    boolean isSlice();

    /**
     *  This method returns the number of slices which have been
     *  created from this very tensor.
     *  It does so by accessing the {@link Relation} component if present
     *  which internally keeps track of slices via weak references.
     *
     * @return The number of slices derived from this tensor.
     */
    int sliceCount();

    /**
     *  If slices have been derived from this tensor then it is a "slice parent".
     *  This is what this method will determine, in which case, it will return true.
     *
     * @return The truth value determining if slices have been derived from this tensor.
     */
    boolean isSliceParent();

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  Detached functions (like those pre-instantiated in Function.Detached.*) will not attach {@link GraphNode}
     *  instances to involved tensors which will prevent the formation of a computation graph.
     *
     * @return The truth value determining if this tensor belongs to a recorded computation graph.
     */
    boolean belongsToGraph();

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  A tensor is a leave if it is attached to a computation graph in which it is not an intermediate / branch node
     *  but input / branch node.
     *
     * @return The truth value determining if this tensor is attached to a computation graph as leave node.
     */
    default boolean isLeave() { return (!this.belongsToGraph() || getGraphNode().isLeave()); }

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  This autograd system however, will only be triggered by {@link Function} implementations which
     *  are not "detached", meaning they have their "{@link Function#isDoingAD()}" flags set to true! <br>
     *  A tensor is a branch if it is attached to a computation graph in which it is not an input / leave node
     *  but intermediate / branch node.
     *
     * @return The truth value determining if this tensor is attached to a computation graph as branch node.
     */
     default boolean isBranch() { return !this.isLeave(); }

    /**
     *  Tensors can be components of other tensors which makes the
     *  implicitly their gradients.
     *
     * @return The truth value determining if this tensor has another tensor attached to it (which is its gradient).
     */
    boolean hasGradient();

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
     *  down to an array holding a single value.
     *  This only makes sense for homogeneously populated tensors.
     *  Passing {@code false} to this method will "actualize" a "virtual" tensor.
     *  Meaning the underlying data array will at least become as large as the size of the tensor
     *  as is defined by {@link #size()}.
     *
     * @param isVirtual The truth value determining if this tensor should be "virtual" or "actual".
     * @return This concrete instance, to allow for method chaining.
     */
    Tsr<V> setIsVirtual( boolean isVirtual );

    /**
     *  This flag works alongside two autograd features which can be enables inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happened when the second flag is set to true as well, because otherwise gradients
     *  wouldn't be applied to their tensors automatically in the first place... <br>
     *  <br>
     *  Setting both flags to true will inhibit the effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
     *  unless a form of "permission" is being signaled to the autograd system.
     *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
     *  be updated by its gradient</b>.<br>
     *  <br>
     * @return The truth value determining if the application of the gradient of this tensor is requested.
     */
    boolean gradientApplyRequested();

    /**
     *  This flag works alongside two autograd features which can be enables inside the library settings.
     *  They will come into effect when flipping their feature flags, <br>
     *  namely: <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
     *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
     *  however this will only happen when the second flag is set to true as well, because otherwise gradients
     *  wouldn't be applied to their tensors automatically in the first place... <br>
     *  <br>
     *  Setting both flags to true will inhibit effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
     *  unless a form of "permission" is being signaled to the autograd system.
     *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
     *  be updated by its gradient</b>.<br>
     *  <br>
     * @param applyRequested The truth value determining if the application of the gradient of this tensor is requested.
     * @return This very tensor instance in order to enable method chaining.
     */
    Tsr<V> setGradientApplyRequested( boolean applyRequested );
    /**
     * @return The type class of individual value items within this {@link Tsr} instance.
     */
    Class<V> getValueClass();

    /**
     * @return The type class of individual value items within this {@link Tsr} instance.
     */
    default Class<V> valueClass() { return getValueClass(); }

    /**
     *  This method returns the {@link DataType} instance of this {@link Tsr}, which is
     *  a wrapper object for the actual type class representing the value items stored inside
     *  the underlying data array of this tensor.
     *
     * @return The {@link DataType} instance of this {@link Tsr} storing important type information.
     */
    DataType<V> getDataType();

    /**
     *  The {@link Class} returned by this method is the representative {@link Class} of the
     *  value items of a concrete {@link AbstractTensor} but not necessarily the actual {@link Class} of
     *  a given value item, this is especially true for numeric types, which are represented by
     *  implementations of the {@link NumericType} interface.                                        <br>
     *  For example in the case of a tensor of type {@link Double}, this method would
     *  return {@link neureka.dtype.custom.F64} which is the representative class of {@link Double}. <br>
     *  Calling the {@link #getValueClass()} method instead of this method would return the actual value
     *  type class, namely: {@link Double}.
     *
     * @return The representative type class of individual value items within this concrete {@link AbstractTensor}
     *         extension instance which might also be sub-classes of the {@link NumericType} interface
     *         to model unsigned types or other JVM foreign numeric concepts.
     */
    Class<?> getRepresentativeValueClass();

    /**
     *  This method compares the passed class with the underlying data-type of this NDArray.
     *  If the data-type of this NDArray is equivalent to the passed class then the returned
     *  boolean will be true, otherwise the method returns false.
     *
     * @param typeClass The class which ought to be compared to the underlying data-type of this NDArray.
     * @return The truth value of the question: Does this NDArray implementation hold the data of the passed type?
     */
    boolean is( Class<?> typeClass );

    /**
     * This method takes a {@link Device} and tries to migrate the contents of this {@link Tsr}
     * instance to that {@link Device}!
     *
     * @param device The {@link Device} which should host this {@link Tsr} as well as be added to its components list.
     * @return This very class to enable method chaining.
     */
    Tsr<V> to( Device<?> device );

    Tsr<V> to( String deviceType );

    /**
     * @return The gradient of this tensor which is internally stored as component.
     */
    Tsr<V> getGradient();

    /**
     * @return The device on which this tensor is stored or {@link CPU} if it is not outsourced.
     */
    Device<V> getDevice();

    /**
     * @return The graph node of the computation graph to which this tensor belongs or null if not part of a graph.
     */
    GraphNode<V> getGraphNode();

    /**
     * @return An instance of the {@link NDFrame} component if present.
     */
    NDFrame<V> frame();

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients.
     *
     * @param error A tensor which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    Tsr<V> backward( Tsr<V> error );

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients.<br>
     *  <br>
     *  This method turns the given scalar value and
     *  turns it into a matching tensor ( with the same shape)
     *  which will then be back-propagated through the
     *  recorded computation graph.
     *
     * @param value A scalar which is back-propagated to gradients. Must match the size og this tensor.
     * @return The tensor on which this method was called. (factory pattern)
     */
    Tsr<V> backward( double value );

    /**
     *  Tensors which are used or produced by the autograd system will have a {@link GraphNode} component attached to them.
     *  This is because autograd requires recording a computation graph for back-prop traversal.
     *  If this tensor is part of a computation graph then this method
     *  will traverse an error backward in the recorded history towards tensors which require
     *  the accumulation of gradients. <br>
     *  <br>
     *  This method assumes that the user wants to back-propagate
     *  an error of "1" having the same shape as
     *  this tensor.
     *
     * @return The tensor on which this method was called. (factory pattern)
     */
    Tsr<V> backward();

    /**
     *  If this tensor owns a gradient tensor as component, then it can be applied by this method. <br>
     *  "Applying" a gradient to a tensor simply means adding the values inside the gradient element-wise
     *  to the owning host tensor via an inline operation. <br>
     */
    void applyGradient();

    /**
     *  <b>This method detaches this tensor from its underlying computation-graph
     *  or simply does nothing if no graph is present.</b> <br>
     *  Nodes within a computation graph are instances of the "{@link GraphNode}" class which are also
     *  simple components of the tensors they represent in the graph. <br>
     *  Therefore, "detaching" this tensor from the graph simply means removing its {@link GraphNode} component.
     *
     * @return This very instance in order to allows for a more streamline usage of this method.
     */
    Tsr<V> detach();

    /**
     *  This method receives a nested {@link String} array which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows:    <br>
     *                                                              <br>
     *      dim 0 : ["A", "B"]                                      <br>
     *      dim 1 : ["1", "2", "3"]                                 <br>
     *                                                              <br>
     *
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String[][] labels );

    /**
     *  This method receives a label for this tensor and a
     *  nested {@link String} array which ought to contain a
     *  label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows:    <br>
     *                                                              <br>
     *      dim 0 : ["A", "B"]                                      <br>
     *      dim 1 : ["1", "2", "3"]                                 <br>
     *                                                              <br>
     *
     * @param tensorName A label for this tensor itself.
     * @param labels A nested String array containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String tensorName, String[][] labels );

    /**
     *  This method receives a nested {@link String} list which
     *  ought to contain a label for the index of this tensor.
     *  The index for a single element of this tensor would be an array
     *  of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *                                                           <br>
     *      dim 0 : ["A", "B"]                                   <br>
     *      dim 1 : ["1", "2", "3"]                              <br>
     *                                                           <br>
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( List<List<Object>> labels );

    /**
     *  This method receives a label for this tensor and a nested
     *  {@link String} list which ought to contain a label for the index of
     *  this tensor The index for a single element of this tensor would
     *  be an array of numbers as long as the rank where every number is
     *  in the range of the corresponding shape dimension...
     *  Labeling an index means that for every dimension there
     *  must be a label for elements in this range array! <br>
     *  For example the shape (2,3) could be labeled as follows: <br>
     *                                                           <br>
     *      dim 0 : ["A", "B"]                                   <br>
     *      dim 1 : ["1", "2", "3"]                              <br>
     *                                                           <br>
     * @param tensorName A label for this tensor itself.
     * @param labels A nested String list containing labels for indexes of the tensor dimensions.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( String tensorName, List<List<Object>> labels );

    /**
     *  This method provides the ability to
     *  label not only the indices of the shape of this tensor, but also
     *  the dimension of the shape.
     *  The first and only argument of the method expects a map instance
     *  where keys are the objects which ought to act as dimension labels
     *  and the values are lists of labels for the indices of said dimensions.
     *  For example the shape (2,3) could be labeled as follows:            <br>
     *  [                                                                   <br>
     *      "dim 0" : ["A", "B"],                                           <br>
     *      "dim 1" : ["1", "2", "3"]                                       <br>
     *  ]                                                                   <br>
     *                                                                      <br>
     * @param labels A map in which the keys are dimension labels and the values are lists of index labels for the dimension.
     * @return This tensor (method chaining).
     */
    Tsr<V> label( Map<Object, List<Object>> labels );

    Tsr<V> label( String tensorName, Map<Object, List<Object>> labels );

    /**
     *  This method will produce the sum of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise addition.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the addition.
     * @return The sum of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> plus( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot add 'null' to a tensor!");
        return Neureka.get().backend().getAutogradFunction().plus().call( (Tsr<V>) this, other );
    }

    default Tsr<V> plusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot add-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().plusAssign().call( (Tsr<V>) this, other );
    }

    /**
     *  This method will create a new {@link Tsr}
     *  with the provided double scalar added to all elements of this {@link Tsr}.
     *
     *  The shapes of this tensor is irrelevant as the provided value will
     *  simply be broadcast to any possible shape.
     *
     * @param value The right operand of the addition.
     * @return The sum between this instance as the left and the passed double as right operand.
     */
    default Tsr<V> plus( V value ) { return plus( Tsr.of( valueClass(), this.shape(), value ) ); }


    /**
     *  This method will perform subtraction on
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise subtraction.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the subtraction.
     * @return The difference between this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> minus( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot subtract 'null' from a tensor!");
        return Neureka.get().backend().getAutogradFunction().minus().call( (Tsr<V>) this, other );
    }

    default Tsr<V> minus( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getValueClass(), "Cannot subtract 'null' from a tensor!");
        return minus(
                Tsr.of( this.getDataType().getValueTypeClass() )
                        .withShape(this.getNDConf().shape())
                        .all(other)
        );
    }
    default Tsr<V> minusAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot subtract-assign 'null' from a tensor!");
        return Neureka.get().backend().getFunction().minusAssign().call( (Tsr<V>) this, other );
    }


    default Tsr<V> minusAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getValueClass(), "Cannot subtract-assign 'null' from a tensor!");
        return minusAssign(
                Tsr.of( this.getDataType().getValueTypeClass() )
                        .withShape(this.getNDConf().shape())
                        .all(other)
        );
    }
    /**
     * @return A clone of this tensor where the signs of all elements are flipped.
     */
    default Tsr<V> negative() { return Neureka.get().backend().getAutogradFunction().neg().call( (Tsr<V>) this ); }

    /**
     *  A method which returns a new {@link Tsr} instance which is a transposed twin of this instance.
     *
     * @return A new transposed tensor with the same underlying data as this tensor.
     */
    default Tsr<V> T() {
        if ( this.rank() == 1 ) return (Tsr<V>) this;
        else if ( this.rank() == 2 ) {
            boolean wasIntermediate = this.isIntermediate();
            this.getUnsafe().setIsIntermediate(false);
            Tsr<V> result = Neureka.get().backend().getFunction().transpose2D().call( (Tsr<V>) this );
            this.getUnsafe().setIsIntermediate(wasIntermediate);
            return result;
        }
        StringBuilder operation = new StringBuilder();
        for ( int i = rank() - 1; i >= 0; i-- ) operation.append( i ).append( i == 0 ? "" : ", " );
        operation = new StringBuilder( "[" + operation + "]:(I[ 0 ])" );
        return Function.of( operation.toString(), true ).call( (Tsr<V>) this );
    }

    /**
     *  This method performs various operations by calling {@link Function} instances
     *  in order to ultimately calculate the mean value of all values
     *  of this very tensor!
     *  This scalar tensor is then returned.
     *
     * @return A scalar tensor which is the mean value of all values of this very tensor.
     */
    default Tsr<V> mean() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> sum = sum();
        Tsr<V> result = functions.div().call( sum, Tsr.of( this.getValueClass(), new int[]{1}, this.size() ) );
        sum.getUnsafe().delete();
        return result;
    }

    default Tsr<V> sum() {
        Functions functions = Neureka.get().backend().getAutogradFunction();
        Tsr<V> ones = Tsr.of( this.getValueClass(), this.getNDConf().shape(), 1 );
        Tsr<V> sum = functions.conv().call( (Tsr<V>) this, ones );
        if ( !ones.has(GraphNode.class) || !ones.getGraphNode().isUsedAsDerivative() )
            ones.getUnsafe().delete();
        if ( sum == null )
            throw new IllegalStateException(
                    "Failed to calculate sum using convolution! Shapes: "+
                            Arrays.toString(this.getNDConf().shape())+"x"+Arrays.toString(ones.getNDConf().shape())
            );
        return sum;
    }

    /**
     *  This method performs a convolutional based dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *
     * @param other The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    default Tsr<V> convDot( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class);
        Tsr<V> a = (Tsr<V>) this;
        int[][] fitter = AbstractTensor.Utility.makeFit( a.getNDConf().shape(), other.getNDConf().shape() );
        boolean doReshape = false;
        for ( int i = 0; i < fitter[ 0 ].length && !doReshape; i++ ) if ( fitter[ 0 ][ i ] != i ) doReshape = true;
        for ( int i = 0; i < fitter[ 1 ].length && !doReshape; i++ ) if ( fitter[ 1 ][ i ] != i ) doReshape = true;
        if ( doReshape ) {
            a = Function.of( AbstractTensor.Utility.shapeString( fitter[ 0 ] ) + ":(I[ 0 ])" ).call( a );
            other = Function.of( AbstractTensor.Utility.shapeString( fitter[ 1 ] ) + ":(I[ 0 ])" ).call( other );
        }
        return Neureka.get()
                .backend()
                .getAutogradFunction()
                .conv()
                .call( a, other )
                .dimtrim();
    }

    /**
     *  This method performs a dot product between the last dimension of this tensor
     *  and the first dimension of the passed tensor.
     *  However, currently this method can only handle matrices which means
     *  that it is functionally completely identical to the {@link #matMul(Tsr)} method.
     *
     * @param other The tensor which is the right part of the dot product operation.
     * @return A new tensor which is the dot product of this tensor and the passed one.
     */
    default Tsr<V> dot( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform dot operation when second operand is 'null'!");
        if ( this.rank() != 2 && other.rank() != 2 )
            throw new IllegalStateException("Not yet implemented!"); // This is not yet available in the backend!
        return this.matMul( other );
    }

    /**
     *  The {@link #matMul(Tsr)} method will produce the matrix product of
     *  two 2 dimensional arrays, where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *
     * @param other The right operand of the matrix multiplication.
     * @return The matrix product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> matMul( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform matrix multiplication operation when second operand is 'null'!");
        if ( this.rank() != 2 || other.rank() != 2 )
            throw new IllegalArgumentException(
                    "Cannot perform matrix multiplication for tensors whose ranks are not both 2!\n" +
                    "Encountered ranks: " + this.rank() + ", " + other.rank() + ";"
                );

        return Neureka.get().backend().getAutogradFunction().matMul().call( (Tsr<V>) this, other );
    }

    /**
     *  This method creates a new tensor sharing the same data and whose shape is trimmed.
     *  A trimmed shape is simply a shape without preceding and trailing ones. <br>
     *  For example the shape (1x4x1x2x1) would be trimmed to (4x1x2).
     *  The underlying operation does not perform a removal of redundant ones all together.
     *  Only ones at the start and the beginning will be removed.
     *  A scalar tensor will not be affected by this operation.
     *
     * @return A tensor with the same underlying data but possibly trimmed shape without preceding or trailing ones.
     */
    default Tsr<V> dimtrim() { return Neureka.get().backend().getAutogradFunction().dimTrim().call( (Tsr<V>) this ); }

    /**
     *  A method which returns a new {@link Tsr} instance which is a transposed twin of this instance.
     *  It is and alias method to the {@link #T()} method...
     *
     * @return A new transposed tensor with the same underlying data as this tensor.
     */
    default Tsr<V> getT() { return this.T(); } // Transposed

    /**
     *  This method name translates to the "in" keyword in Groovy!
     *  The same is true for the "contains" method in Kotlin.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    boolean isCase( Tsr<V> other );

    /**
     *  This method name translates to the "in" keyword in Kotlin!
     *  The same is true for the "isCase" method in Groovy.
     *  Both methods do the exact same thing, however they exist
     *  for better language support.
     *
     * @param other The tensor which will be checked.
     * @return The answer to the following question: Is the data of the provided tensor a subset of the data of this tensor?
     */
    default boolean contains( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform 'contains' operation when second operand is 'null'!");
        return this.isCase( other );
    }

    /**
     *  This is technically the equivalent to a full slice.
     *
     * @return A shallow copy where the underlying data is shared with this tensor.
     */
    default Tsr<V> shallowCopy() {
        if ( this.isEmpty() || this.isUndefined() ) return (Tsr<V>) this;
        List<List<Integer>> ranges = new ArrayList<>();
        for ( int e : this.shape() ) {
            List<Integer> rangeAsList = new ArrayList<>();
            for ( int i = 0; i < e; i++ ) rangeAsList.add( i );
            ranges.add( rangeAsList);
        }
        return getAt( ranges.toArray() );
    }

    /**
     *  This method returns a {@link SliceBuilder} instance exposing a simple builder API
     *  which enables the configuration of a slice of the current tensor via method chaining.    <br>
     *  The following code snippet slices a 3-dimensional tensor into a tensor of shape (2x1x3)  <br>
     * <pre>{@code
     *  myTensor.slice()
     *          .axis(0).from(0).to(1)
     *          .then()
     *          .axis(1).at(5) // equivalent to '.from(5).to(5)'
     *          .then()
     *          .axis().from(0).to(2)
     *          .get();
     * }</pre>
     *
     * @return An instance of the {@link SliceBuilder} class exposing a readable builder API for creating slices.
     */
    SliceBuilder<V> slice();

    /**
     *  This method is synonymous to the {@link #times(Tsr)} method.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise product.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> multiply( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply 'null' with a tensor!");
        return Neureka.get().backend().getAutogradFunction().mul().call( (Tsr<V>) this, other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> multiply( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getValueClass(), "Cannot multiply 'null' with a tensor!");
        return multiply(
                Tsr.of( this.getDataType().getValueTypeClass() )
                        .withShape( this.getNDConf().shape() )
                        .all( other )
        );
    }

    /**
     *  The {@link #times(Tsr)} method is synonymous to the {@link #multiply(Tsr)}.
     *  Both of which will produce the product of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise product.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the multiplication.
     * @return The product of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> times( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }

    /**
     * @param other The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> times( V other ) {
        LogUtil.nullArgCheck(other, "other", getValueClass(), "Cannot multiply 'null' with a tensor!");
        return multiply( other );
    }
    /**
     * @param other The tensor whose elements ought to be multiplied and assigned to elements in this tensor.
     * @return This instance where each value element was multiplied by the corresponding element in the provided tensor.
     */
    default Tsr<V> timesAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot multiply-assign 'null' to a tensor!");
        return Neureka.get().backend().getFunction().mulAssign().call( (Tsr<V>) this, other );
    }

    /**
     * @param other The value which ought to be multiplied and assigned to each element in this tensor.
     * @return This instance where each value element was multiplied by the provided element.
     */
    default Tsr<V> timesAssign( V other ) {
        LogUtil.nullArgCheck(other, "other", this.getValueClass(), "Cannot multiply-assign 'null' to a tensor!");
        return this.timesAssign( Tsr.of( getValueClass(), getNDConf().shape(), other ) );
    }
    /**
     * @param value The value which should be broadcast to all elements of a clone of this tensor.
     * @return A new clone of this tensor where all elements are multiplied by the provided value.
     */
    default Tsr<V> multiply( double value ) { return multiply( Tsr.of( getValueClass(), getNDConf().shape(), value ) ); }

    /**
     *  The {@link #div(Tsr)} method will produce the quotient of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise division.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the division.
     * @return The quotient of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> div( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot divide a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getAutogradFunction().div().call( (Tsr<V>) this, other );
    }
    default Tsr<V> div( V value ) { return div( Tsr.of( getValueClass(), getNDConf().shape(), value ) ); }

    default Tsr<V> divAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot divide-assign a tensor by 'null' (In any sense of the word)!");
        return Neureka.get().backend().getFunction().divAssign().call( (Tsr<V>) this, other );
    }

    /**
     *  The {@link #mod(Tsr)} method will produce the modulus of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise modulo operation.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand of the modulo operation.
     * @return The modulus of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> mod( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getAutogradFunction().mod().call( (Tsr<V>) this, other );
    }

    default Tsr<V> mod( int other ) { return mod(Tsr.of(getValueClass(), getNDConf().shape(), other)); }

    /**
     *  This method is synonymous to the {@link #mod(int)} method.
     */
    default Tsr<V> rem( int other ) { return this.mod(other); }

    default Tsr<V> modAssign( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot perform tensor modulo 'null'!");
        return Neureka.get().backend().getFunction().modAssign().call( (Tsr<V>) this, other );
    }

    /**
     *  The {@link #power(Tsr)} (Tsr)} method will produce the power of
     *  two tensors with the same rank (or two ranks which can be made compatible with padding ones),
     *  where the left operand is this {@link Tsr}
     *  instance and the right operand is the tensor passed to the method.
     *  If the shapes of both of the involved tensors is identical then
     *  the result will be a regular element-wise exponentiation.
     *  Otherwise, the method will also be able to perform broadcasting, however only if
     *  for every pair of shape dimension the following is true:
     *  Either the dimensions have the same size or one of them has size 1. <br>
     *  Here is an example of 2 matching shapes: (1, 4, 1) and (3, 4, 1)       <br>
     *  And here is an example of a mismatch: (2, 4, 1) and (3, 4, 1)         <br>
     *
     * @param other The right operand, also known as exponent, of the exponentiation.
     * @return The power of this instance as the left and the passed {@link Tsr} instance as right operand.
     */
    default Tsr<V> power( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( (Tsr<V>) this, other );
    }

    default Tsr<V> power( V value ) {
        return power( Tsr.of( this.valueClass(), this.shape(), value ) );
    }

    /**
     *  This method is synonymous to the {@link #power(Tsr)} method.
     */
    default Tsr<V> xor( Tsr<V> other ) {
        LogUtil.nullArgCheck(other, "other", Tsr.class, "Cannot raise a tensor to the power of 'null'!");
        return Neureka.get().backend().getAutogradFunction().pow().call( (Tsr<V>) this, other );
    }

    /**
     *  This method is synonymous to the {@link #power(Tsr)} method.
     */
    default Tsr<V> xor( double value ) { return xor( Tsr.of( this.valueClass(), this.shape(), value ) ); }

    /**
     *  Settings this flag via this setter will indirectly trigger the activation of
     *  the autograd / auto-differentiation system of this library!
     *  If the flag is set to 'true' and the tensor is used for computation then
     *  it will also receive gradients when the {@link #backward()} method is being called
     *  on any descendant tensor within the computation graph.
     *
     * @param rqsGradient The truth value determining if this tensor ought to receive gradients via
     *                     the built-in automatic backpropagation system.
     * @return This very {@link Tsr} instance in order to enable method chaining.
     */
    Tsr<V> setRqsGradient( boolean rqsGradient );

    /**
     *  This flag will indirectly trigger the activation of the autograd / auto-differentiation system of this library!
     *  If the flag is set to 'true' and the tensor is used for computation then
     *  it will also receive gradients when the {@link #backward()} method is being called
     *  on any descendant tensor within the computation graph.
     *
     * @return The truth value determining if this tensor ought to receive gradients via
     *         the built-in automatic backpropagation system.
     */
    boolean rqsGradient();

    /**
     *  Intermediate tensors are internal non-user tensors which may be eligible
     *  for deletion when further consumed by a {@link Function}.
     *  For the casual user of Neureka, this flag should always be false!
     *
     * @return The truth value determining if this tensor is not a user tensor but an internal
     *         tensor which may be eligible for deletion by {@link Function}s consuming it.
     */
    boolean isIntermediate();

    /**
     *  This method informs this tensor if it's data is supposed to be kept in RAM
     *  or if it has already been migrated somewhere else.
     *  In the latter case, the tensor will nullify the reference to it's
     *  underlying data array to make it elegable for garbage collection.
     *  Otherwise, if {@code isOutsourced} is set to true, the method might
     *  allocate a new data array if none is present.
     *
     * @param isOutsourced The truth value which determines if this tensor should live in RAM or somewhere else.
     * @return This very instance to allow for method chaining.
     */
    Tsr<V> setIsOutsourced( boolean isOutsourced );

    /**
     *  Outsourced means that the tensor is stored on a {@link Device} implementation instance.
     *
     * @return The truth value determining if the data of this tensor is not actually stored inside of it
     *         in the form of of a traditional primitive JVM array!
     */
    boolean isOutsourced();

    /**
     *  This method exposes an API for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptional situations!<br>
     *  <br><b>
     *
     *  Only use this if you know what you are doing and
     *  performance is critical! <br>
     *  </b>
     *  (Like custom backend extensions for example)
     *
     * @return The unsafe API exposes methods for mutating the state of the tensor.
     */
    Unsafe<V> getUnsafe();


    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    Tsr<V> getAt( int... indices );

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> getAt( Number i ) {
        return getAt( Collections.singletonList( getNDConf().indicesOfIndex( (i).intValue() ) ).toArray() );
    }

    /**
     *  The following method enables access to specific scalar elements within the tensor.
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param indices The index array of the element which should be returned.
     * @return An element located at the provided index.
     */
    default Tsr<V> get( int... indices ) { return getAt( indices ); }

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    default Tsr<V> getAt( Object... args ) {
        List<Object> argsList = Arrays.asList( args );
        return getAt( argsList );
    }

    /**
     *  The following method enables the creation of tensor slices which access
     *  the same underlying data (possibly from a different view).
     *  The method name also translates to the subscript operator in Groovy.
     *
     * @param args An arbitrary number of arguments which can be used for slicing.
     * @return A slice tensor created based on the passed keys.
     */
    default Tsr<V> get( Object... args ) {
        return getAt( args );
    }

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> getAt( int i ) { return getAt( indicesOfIndex(i) ); }

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> get( int i ) { return getAt( i ); }

    /**
     *  This getter method creates and returns a slice of the original tensor.
     *  The returned slice is a scalar tensor wrapping a single value element which
     *  is being targeted by the provided integer index.
     *
     * @param i The index of the value item which should be returned as a tensor instance.
     * @return A tensor holding a single value element which is internally still residing in the original tensor.
     */
    default Tsr<V> get( Number i ) { return getAt( i ); }

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    default Tsr<V> get( Object key ) { return getAt( key ); }

    /**
     *  This method is most useful when used in Groovy
     *  where defining maps is done through square brackets,
     *  making it possible to slice tensors like so: <br>
     *  <pre>{@code
     *      var b = a[[[0..0]:1, [0..0]:1, [0..3]:2]]
     *  }</pre>
     *  Here a single argument with the format '[i..j]:k' is equivalent
     *  to Pythons 'i:j:k' syntax for indexing! (numpy)                            <br>
     *  i... start indexAlias.                                                      <br>
     *  j... end indexAlias. (inclusive!)                                           <br>
     *  k... step size.
     *
     * @param rangToStrides A map where the keys define where axes should be sliced and values which define the strides for the specific axis.
     * @return A tensor slice with an offset based on the provided map keys and
     *         strides based on the provided map values.
     */
    Tsr<V> getAt( Map<?,Integer> rangToStrides );

    /**
     *  This method enables tensor slicing!
     *  It takes a key of various types and configures a slice
     *  tensor which shares the same underlying data as the original tensor.
     *
     * @param key This object might be a wide range of objects including maps, lists or arrays...
     * @return A slice tensor or scalar value.
     */
    Tsr<V> getAt( List<?> key );

    /**
     *  This method enables assigning a provided tensor to be a subset/slice of this tensor!
     *  It takes a key which is used to configure a slice
     *  sharing the same underlying data as the original tensor.
     *  This slice is then used to assign the second argument {@code value} to it.
     *  The usage of this method is especially powerful when used in Groovy. <br>
     *  The following code illustrates this very well:
     *  <pre>{@code
     *      a[[[0..0]:1, [0..0]:1, [0..3]:2]] = b
     *  }</pre>
     *  Here a single argument with the format '[i..j]:k' is equivalent
     *  to pythons 'i:j:k' syntax for indexing! (numpy)                            <br>
     *  i... start indexAlias.                                                      <br>
     *  j... end indexAlias. (inclusive!)                                           <br>
     *  k... step size.                                                             <br>
     *
     * @param key This object is a map defining a stride and a targeted index or range of indices...
     * @param value The tensor which ought to be assigned into a slice of this tensor.
     * @return A slice tensor or scalar value.
     */
    Tsr<V> putAt( Map<?,Integer> key, Tsr<V> value );


    Tsr<V> putAt( int[] indices, V value );

    /**
     *  Use this to place a single item at a particular position within this tensor!
     *
     * @param indices An array of indices targeting a particular position in this tensor...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    default Tsr<V> set( int[] indices, V value ) {
        return putAt( indices, value );
    }


    /**
     *  Individual entries for value items in this tensor can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this tensor
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    default Tsr<V> putAt( int index, V value ) { return putAt( indicesOfIndex(index), value ); }


    /**
     *  Individual entries for value items in this tensor can be set
     *  via this method.
     *
     * @param index The scalar index targeting a specific value position within this tensor
     *          which ought to be replaced by the one provided by the second parameter
     *          of this method.
     *
     * @param value The item which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    default Tsr<V> set( int index, V value ) { return putAt( index, value ); }

    /**
     *  This method enables injecting slices of tensor to be assigned into this tensor!
     *  It takes a key of various types which is used to configure a slice
     *  tensor sharing the same underlying data as the original tensor.
     *  This slice is then used to assign the second argument to it, namely
     *  the "value" argument.
     *
     * @param key This object is a list defining a targeted index or range of indices...
     * @param value the tensor which ought to be assigned to a slice of this tensor.
     * @return This very tensor in order to enable method chaining...
     */
    Tsr<V> putAt( List<?> key, Tsr<V> value );

    /**
     *  Use this to place a single item at a particular position within this tensor!
     *
     * @param indices A list of indices targeting a particular position in this tensor...
     * @param value the value which ought to be placed at the targeted position.
     * @return This very tensor in order to enable method chaining...
     */
    default Tsr<V> putAt( List<?> indices, V value ) {
        return this.putAt( indices, Tsr.of( this.getValueClass(), shape(), value ) );
    }

    /**
     *  An NDArray implementation ought to have some way to access its underlying data array.
     *  This method simple returns an element within this data array sitting at position "i".
     * @param i The position of the targeted item within the raw data array of an NDArray implementation.
     * @return The found object sitting at the specified index position.
     */
    V getDataAt( int i );

    /**
     *  An NDArray implementation ought to have some way to selectively modify its underlying value.
     *  This method simply overrides an element within this data array sitting at position "i".
     * @param i The index of the value array entry which ought to be addressed.
     * @param o The object which ought to be placed at the requested position.
     * @return This very tensor in order to enable method chaining.
     */
    Tsr<V> setValueAt( int i, V o );

    /**
     *  The following method returns a raw value item within this tensor
     *  targeted by a scalar index.
     *
     * @param i The scalar index of the value item which should be returned by the method.
     * @return The value item found at the targeted index.
     */
    default V getValueAt( int i ) { return getDataAt( indexOfIndex( i ) ); }

    /**
     *  This method returns a raw value item within this tensor
     *  targeted by an index array which is expect to hold an index for
     *  every dimension of the shape of this tensor.
     *  So the provided array must have the same length as the
     *  rank of this tensor!
     *
     * @param indices The index array which targets a single value item within this tensor.
     * @return The found raw value item targeted by the provided index array.
     */
    default V getValueAt( int... indices ) {
        LogUtil.nullArgCheck( indices, "indices", int[].class, "Cannot find tensor value without indices!" );
        if ( indices.length == 0 ) throw new IllegalArgumentException("Index array may not be empty!");
        if ( indices.length < this.rank() ) {
            if ( indices.length == 1 ) return getDataAt( getNDConf().indexOfIndex( indices[0] ) );
            else {
                int[] allIndices = new int[this.rank()];
                System.arraycopy( indices, 0, allIndices, 0, indices.length );
                return getDataAt( getNDConf().indexOfIndices( allIndices ) );
            }
        }
        return getDataAt( getNDConf().indexOfIndices( indices ) );
    }

    /**
     *  This method will receive an object an try to interpret
     *  it or its contents to be set as value for this tensor.
     *  It will not necessarily replace the underlying data array object of this
     *  tensor itself, but also try to convert and copy the provided value
     *  into the data array of this tensor.
     *
     * @param value The value which may be a scalar or array and will be used to populate this tensor.
     * @return This very tensor to enable method chaining.
     */
    Tsr<V> setValue( Object value );

    Object getValue();

    /**
     *  This returns an unprocessed version of the underlying data of this tensor.
     *  If this tensor is outsourced (stored on a device), then the data will be loaded
     *  into an array and returned by this method.
     *  Do not expect the returned array to be actually stored within the tensor itself!
     *  Contrary to the {@link Tsr#getValue()} method, this one will
     *  return the data in an unbiased form, where for example a virtual (see {@link #isVirtual()})
     *  tensor will have this method return an array of length 1.
     *
     * @return An unbiased copy of the underlying data of this tensor.
     */
    Object getData();

    <T> Tsr<T> mapTo(
            Class<T> typeClass,
            java.util.function.Function<V,T> mapper
    );

    /**
     *  Turns this tensor into a {@link BufferedImage} based on the provided
     *  {@link Tsr.ImageType} formatting choice.
     *
     * @param type The type of format used to create the buffered image.
     * @return A {@link BufferedImage} populated with the contents of this tensor.
     */
    BufferedImage asImage( Tsr.ImageType type );

    /**
     *  This method takes the provided {@link Tsr} instance and adds its
     *  contents to the contents of the {@link Tsr} which is set as gradient of this very {@link Tsr}.
     *
     * @param error The error gradient which ought to be added to the gradient of this tensor.
     * @return This very tensor instance to enable method chaining.
     */
    Tsr<V> addToGradient( Tsr<V> error );

    /**
     * @param typeClass The class which is the target of the type conversion.
     * @param <T> The type parameter of the type that will be returned.
     * @return An instance of the supplied type class.
     */
    public <T> T asType( Class<T> typeClass );

    default  <A> A getValueAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getValue(), arrayTypeClass );
    }

    default  <A> A getDataAs( Class<A> arrayTypeClass ) {
        return DataConverter.get().convert( getData(), arrayTypeClass );
    }

    default String toString( String conf ) {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( (Tsr<?>) this ).withConfig( conf ).toString();
    }

    default String toString( TsrStringSettings config ) {
        if ( this.isDeleted() ) return "deleted";
        else if ( this.isEmpty() ) return "empty";
        else if ( this.isUndefined() ) return "undefined";
        return TsrAsString.representing( (Tsr<?>) this ).withConfig( config ).toString();
    }

    /**
     *  This allows you to provide a lambda to configure how this tensor should be
     *  converted to {@link String} instances.
     *  The provided {@link Consumer} will receive a {@link TsrStringSettings} instance
     *  which allows you to change various settings with the help of method chaining.
     *
     * @param config A consumer of the {@link TsrStringSettings} ready to be configured.
     * @return The {@link String} representation of this tensor.
     */
    default String toString( Consumer<TsrStringSettings> config ) {
        if ( this.isDeleted() ) return "deleted";
        TsrStringSettings defaults = Neureka.get().settings().view().getTensorSettings().clone();
        config.accept(defaults);
        return TsrAsString.representing( (Tsr<?>) this ).withConfig( defaults ).toString();
    }

    String toString();

    /**
     *  The version number is tracking how often this tensor has been mutated.
     *  This is especially useful for checking the correcting of autp-grad!
     */
    int getVersion();

    default Access<V> at( int... indices ) {
        return new Access<V>() {
            @Override public V get() { return getValueAt( indices ); }

            @Override public void set( V value ) { putAt( indices, value ); }
        };
    }

    interface Access<V> {

        V get();

        void set( V value );

    }

    enum ImageType
    {
        RGB_1INT(1, UI32.class, 1),
        ARGB_1INT(2, UI32.class, 1),
        ARGB_PRE_1INT(3, UI32.class, 1),
        BGR_1INT(4, UI32.class, 1),
        BGR_3BYTE(5, UI8.class, 3),
        ABGR_4BYTE(6, UI8.class, 4),
        ABGR_PRE_4BYTE(7, UI8.class, 4),
        RGB_565_USHORT(8, UI16.class, 1),
        RGB_555_USHORT(9, UI16.class, 1),
        GRAY_BYTE(0, UI8.class, 1),
        GRAY_USHORT(1, UI16.class, 1);

        public final int bufferType;
        public final DataType<?> dataType;
        public final int numberOfChannels;

        ImageType( int bufferType, Class<?> valueTypeClass, int numberOfChannels ) {
            this.bufferType = bufferType;
            this.dataType = DataType.of( valueTypeClass );
            this.numberOfChannels = numberOfChannels;
        }
    }

    /**
     *  Tensors should be considered immutable, however sometimes it
     *  is important to mutate their state for performance reasons.
     *  This interface exposes several methods for mutating the state of this tensor.
     *  The usage of methods exposed by this API is generally discouraged
     *  because the exposed state can easily lead to broken tensors and exceptions...<br>
     *  <br>
     */
    interface Unsafe<T> {
        /**
         *  This method sets the NDConfiguration of this NDArray.
         *  Therefore, it should not be used lightly as it can cause major internal inconsistencies.
         *
         * @param configuration The new NDConfiguration instance which ought to be set.
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> setNDConf( NDConfiguration configuration );
        /**
         *  This method is an inline operation which changes the underlying data of this tensor.
         *  It converts the data types of the elements of this tensor to the specified type!<br>
         *  <br>
         *  <b>WARNING : The usage of this method is discouraged for the following reasons: </b><br>
         *  <br>
         *  1. Inline operations are inherently error-prone for most use cases. <br>
         *  2. This inline operation in particular has no safety net,
         *     meaning that there is no implementation of version mismatch detection
         *     like there is for those operations present in the standard operation backend...
         *     No exceptions will be thrown during backpropagation! <br>
         *  3. This method has not yet been implemented to also handle instances which
         *     are slices of parent tensors!
         *     Therefore, there might be unexpected performance penalties or side effects
         *     associated with this method.<br>
         *     <br>
         *
         * @param typeClass The target type class for elements of this tensor.
         * @param <V> The type parameter for the returned tensor.
         * @return The same tensor instance whose data has been converted to hold a different type.
         */
        <V> Tsr<V> toType( Class<V> typeClass );

        /**
         *  Use this to do a runtime checked upcast of the type parameter of the tensor.
         *  This is unsafe because it is in conflict with the {@link #valueClass()}
         *  method.
         *
         * @param superType The class of the super type of the tensor's value type.
         * @return A tensor whose type parameter is upcast.
         * @param <U> The super type parameter of the value type of the tensor.
         */
        <U/*super T*/> Tsr<U> upcast(Class<U> superType );

        /**
         *  This method enables modifying the data-type configuration of this {@link AbstractTensor}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable inconsistencies between the
         *  underlying {@link DataType} instance of this {@link AbstractTensor} and the actual type of the actual
         *  data it is wrapping (or it is referencing on a {@link neureka.devices.Device}).<br>
         *  <br>
         * @param dataType The new {@link DataType} which ought to be set.
         * @return The tensor with the new data type set.
         */
        <V> Tsr<V> setDataType( DataType<V> dataType );

        /**
         *  This method allows you to modify the data-layout of this {@link AbstractTensor}.
         *  Warning! The method should not be used unless absolutely necessary.
         *  This is because it can cause unpredictable side effects especially for certain
         *  operations expecting a particular data layout (like for example matrix multiplication).
         *  <br>
         *
         * @param layout The layout of the data array (row or column major).
         * @return The final instance type of this class which enables method chaining.
         */
        Tsr<T> toLayout( NDConfiguration.Layout layout );

        /**
         *  This method is responsible for incrementing
         *  the "_version" field variable which represents the version of the data of this tensor.
         *  Meaning :
         *  Every time the underlying data (_value) changes this version ought to increment alongside.
         *  The method is called during the execution procedure.
         *
         * @param call The context object containing all relevant information that defines a call for tensor execution.
         * @return This very tensor instance. (factory pattern)
         */
        Tsr<T> incrementVersion( ExecutionCall<?> call );

        /**
         *  Intermediate tensors are internal non-user tensors which may be eligible
         *  for deletion when further consumed by a {@link Function}.
         *  For the casual user of Neureka, this flag should always be false!
         *
         * @param isIntermediate The truth value determining if this tensor is not a user tensor but an internal
         *                       tensor which may be eligible for deletion by {@link Function}s consuming it.
         * @return The tensor to which this unsafe API belongs.
         */
        Tsr<T> setIsIntermediate( boolean isIntermediate );

        /**
         *  Although tensors will be garbage collected when they are not strongly referenced,
         *  there is also the option to manually free up the tensor and its associated data in a native environment.
         *  This is especially useful when tensors are stored on a device like the {@link neureka.devices.opencl.OpenCLDevice}.
         *  In that case calling this method will free the memory reserved for this tensor on the device.
         *  This manual memory freeing through this method can be faster than waiting for
         *  the garbage collector to kick in at a latr point in time... <br>
         *  <br>
         *
         * @return The tensor wo which this unsafe API belongs to allow for method chaining.
         */
        Tsr<T> delete();

        /**
         *  This returns the underlying raw data object of this tensor.
         *  Contrary to the {@link Tsr#getValue()} ()} method, this one will
         *  return an unbiased view on the raw data of this tensor.
         *  Be careful using this, as it exposes mutable state!
         *
         * @return The raw data object underlying this tensor.
         */
        Object getData();

        <A> A getDataAs( Class<A> arrayTypeClass );

        /**
         *  A tensor ought to have some way to selectively modify its underlying data array.
         *  This method simply overrides an element within this data array sitting at position "i".
         * @param i The index of the data array entry which ought to be addressed.
         * @param o The object which ought to be placed at the requested position.
         * @return This very tensor in order to enable method chaining.
         */
        Tsr<T> setDataAt( int i, T o );

        /**
         *  Use this to access the underlying writable data of this tensor if
         *  you want to modify it.
         *  This method will ensure that you receive an instance of whatever array type you provide
         *  or throw descriptive exceptions to make sure that any unwanted behaviour does not
         *  spread further in the backend.
         *
         * @param arrayTypeClass The expected array type underlying the tensor.
         * @param <A> The type parameter of the provided type class.
         * @return The underlying data array of this tensor.
         */
        default <A> A getDataForWriting( Class<A> arrayTypeClass ) {
            LogUtil.nullArgCheck( arrayTypeClass, "arrayTypeClass", Class.class, "Array type must not be null!" );
            if ( !arrayTypeClass.isArray() )
                throw new IllegalArgumentException("Provided type is not an array type.");
            Object data = Unsafe.this.getData();
            if ( data == null )
                throw new IllegalStateException("Could not find writable tensor data for this tensor (Maybe this tensor is stored on a device?).");

            if ( !arrayTypeClass.isAssignableFrom(data.getClass()) )
                throw new IllegalStateException("The data of this tensor does not match the expect type! Expected '"+arrayTypeClass+"' but got '"+data.getClass()+"'.");

            return (A) data;
        }

    }

}
