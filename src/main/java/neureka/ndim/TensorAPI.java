package neureka.ndim;

import neureka.Neureka;
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.ExecutionCall;
import neureka.calculus.Function;
import neureka.common.utility.LogUtil;
import neureka.devices.Device;
import neureka.devices.host.CPU;
import neureka.dtype.DataType;
import neureka.dtype.NumericType;
import neureka.framing.NDFrame;
import neureka.framing.Relation;
import neureka.ndim.config.NDConfiguration;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

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
