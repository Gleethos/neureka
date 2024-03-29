/*
MIT License

Copyright (c) 2019 Gleethos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      _____      _       _   _
     |  __ \    | |     | | (_)
     | |__) |___| | __ _| |_ _  ___  _ __
     |  _  // _ \ |/ _` | __| |/ _ \| '_ \
     | | \ \  __/ | (_| | |_| | (_) | | | |
     |_|  \_\___|_|\__,_|\__|_|\___/|_| |_|

     A tensor component relating slices with their parent tensors.

*/

package neureka.framing;

import neureka.Tensor;
import neureka.common.composition.Component;

import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

/**
 *  This class is an important tensor component responsible for
 *  managing the relationships between slices and the tensors from which
 *  they have been derived.
 *  In case a tensor is a slice then it will have a {@link Relation} instance as
 *  component which will reference the parent tensor strongly (so that its data will not be lost).
 *  However, in case a tensor is the "parent" of a slice tensor then it will
 *  contain a {@link Relation} instance which references the slices weakly (so that they can be garbage collected). <br>
 *  <br>
 *
 * @param <V> The data type class of the elements of the tensor to which this Relation belongs to.
 */
public class Relation<V> implements Component<Tensor<V>>
{
    /**
     *  If the tensor owning this {@link Relation} component
     *  is in fact a slice, then the following field will reference
     *  the tensor from which the slice has been created. <br>
     *  This "parent" tensor is referenced strongly so that the
     *  data which is shared by its slice(s) cannot be garbage collected. <br>
     *  One might wonder: "Why is the data not referenced here directly"? <br>
     *  The answer is simple: The parent tensor (as well as its slices)
     *  might be outsourced to a device which may store the data itself in various ways...
     */
    private Tensor<V> _parent; // Children need their parents. They shall not be garbage collected.

    /**
     *  This is an array of the weakly referenced slice children of the tensor
     *  to which this Relation component belongs. <br>
     *  Children are not referenced strongly so they can be garbage collected.
     */
    private WeakReference<Tensor<V>>[] _children; // Children may be garbage collected if not needed anywhere.

    /**
     *  When creating permuted versions of slices then
     *  there must be a translation between the shape configuration between
     *  this new slice and the original parent tensor from which both slices
     *  have been derived. <br>
     *  This translation, referred to as 'strides', is in essence merely an int array which
     *  contains the index mapping to a new shape.
     *  When accessing data for a permuted slice then this
     *  translation will be necessary for getting the right data. <br>
     *  <br>
     *  This field variable stores an array of int arrays which represent
     *  dimension order mappings for permuted slice tensors of the tensor
     *  to which this Relation instance is a component. <br>
     */
    private int[][] _shapeRelations;

    public static <T> Relation<T> newParentToChildren() {
        return new Relation<>( null );
    }

    public static <T> Relation<T> newChildToParent( Tensor<T> parent ) {
        return new Relation<>( parent );
    }

    private Relation( Tensor<V> parent ) { _parent = parent; }

    @Override
    public boolean update( OwnerChangeRequest<Tensor<V>> changeRequest ) {
        Tensor<V> oldOwner = changeRequest.getOldOwner();
        Tensor<V> newOwner = changeRequest.getNewOwner();
        if ( changeRequest.type() == IsBeing.ADDED || changeRequest.type() == IsBeing.REMOVED ) {
            changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
            return true; // Initial/last update call: No action needed!
        }
        if ( _parent != null) {
            Relation<V> pr = _parent.get( Relation.class );
            for ( int i = 0; i < pr._children.length; i++ ) {
                if ( pr._children[ i ].get() == oldOwner ) {
                    pr._children[ i ] = new WeakReference<>(newOwner);
                }
            }
        }
        if ( _children != null ) {
            for ( WeakReference<Tensor<V>> c : _children ) {
                Tensor<V> t = c.get();
                if ( t != null ) {
                    Relation<V> cr = (Relation<V>) t.get( Relation.class );
                    if ( cr != null ) cr._parent = newOwner;
                }
            }
        }
        changeRequest.executeChange(); // This can be an 'add', 'remove' or 'transfer' of this component!
        return true;
    }

    public Relation<V> addChild( Tensor<V> child )
    {
        if ( _children == null ) {
            _children = new WeakReference[]{ new WeakReference<>( child ) };
            _shapeRelations = new int[ 1 ][];
        } else {
            WeakReference<Tensor<V>>[] newChildren = new WeakReference[ _children.length + 1 ];
            int[][] newShapeRelations = new int[ _children.length + 1 ][];
            System.arraycopy( _children, 0, newChildren, 0, _children.length );
            System.arraycopy( _shapeRelations, 0, newShapeRelations, 0, _children.length );
            newChildren[_children.length] = new WeakReference( child );
            newShapeRelations[_children.length] = null;
            _children = newChildren;
            _shapeRelations = newShapeRelations;
        }
        return this;
    }

    /**
     * When creating permuted versions of slices then
     * there must be a translation between the shape configuration between
     * this new slice and the original parent tensor from which both slices
     * have been derived. <br>
     * This translation is in essence merely an int array which
     * contains the index mapping to a new shape.
     * When accessing data for a permuted slice then this
     * translation will be necessary for getting the right data. <br>
     * <br>
     * This method enables adding such a permute translation associated
     * to a slice, which is also the "child" of the tensor to which this
     * Relation component belongs! <br>
     * <br>
     *
     * @param child   The child (slice) tensor which has a shape whose dimensions are in a different order.
     * @param permuteOrder The int array defining the axis order (dimension index mapping).
     */
    public void addPermuteRelationFor(Tensor<V> child, int[] permuteOrder ) {
        for ( int i = 0; i < _shapeRelations.length; i++ ) {
            Tensor<V> c = _children[ i ].get();
            if ( c != null && c == child )
                _shapeRelations[ i ] = permuteOrder;
        }
    }

    /**
     *  When creating permuted versions of slices then
     *  there must be a translation between the shape configuration between
     *  this new slice and the original parent tensor from which both slices
     *  have been derived. <br>
     *  This translation is in essence merely an int array which
     *  contains the index mapping to a new shape.
     *  When accessing data for a permuted slice then this
     *  translation will be necessary for getting the right data. <br>
     *  <br>
     *  This method can be used to access the dimension order translation (permute)
     *  from the order of the parent tensor (which is the component owner of this Relation)
     *  and the passed slice (which is a weakly referenced child tensor...). <br>
     *  <br>
     *
     * @param child The child (slice) tensor which has a shape whose dimensions are in a different order.
     * @return The int array defining the reshaping (dimension index mapping).
     */
    public int[] getPermuteRelationFor( Tensor<V> child )
    {
        for ( int i = 0; i < _shapeRelations.length; i++ ) {
            Tensor<V> c = _children[ i ].get();
            if ( c != null && c == child )
                return _shapeRelations[ i ];
        }
        return null;
    }

    public List<Tensor<?>> getChildren() {
        List<Tensor<?>> children = new ArrayList<>();
        if ( _children != null ) {
            for ( WeakReference<Tensor<V>> r : _children ) {
                Tensor<V> c = r.get();
                if ( c != null ) {
                    children.add(c);
                    Relation<V> relation = (Relation<V>) c.get( Relation.class );
                    if ( relation != null ) children.addAll(relation.getChildren());
                }
            }
        }
        return children;
    }

    /**
     *  This method tries to find the root data ancestor of this tensor.
     *  If this tensor is not a slice of another tensor, then it can not have data parents
     *  and therefore also not a root tensor, in which case the method will return null!
     *
     * @return The root data parent which actually owns the data of the sliced data or null if the tensor is not a slice.
     */
    public Optional<Tensor<V>> findRootTensor()
    {
        if ( _parent == null ) return Optional.empty();
        else if ( !_parent.has( Relation.class ) ) return Optional.empty();
        else if ( !_parent.get( Relation.class ).hasParent() ) return Optional.of(_parent);
        else return _parent.get( Relation.class ).findRootTensor();
    }

    public boolean hasParent()
    {
        return _parent != null;
    }

    public boolean hasChildren()
    {
        return _children != null && _children.length > 0;
    }

    public int childCount()
    {
        return ( _children == null ? 0 : (int) Arrays.stream(_children).filter( c -> c.get() != null ).count() );
    }

    public void removeChild( Tensor<V> child )
    {
        if ( _children == null ) return;
        int found = -1;
        for ( int i = 0; i < _children.length; i++ )
            if ( _children[i].get() == child ) {
                found = i;
                break;
            }

        if ( found >= 0 ) {
            if ( _children.length == 1 ) {
                _children = null;
                _shapeRelations = null;
            } else {
                WeakReference<Tensor<V>>[] newChildren = new WeakReference[ _children.length - 1 ];
                int[][] newShapeRelations = new int[ _children.length - 1 ][];
                System.arraycopy( _children, 0, newChildren, 0, found );
                System.arraycopy( _shapeRelations, 0, newShapeRelations, 0, found );
                System.arraycopy( _children, found + 1, newChildren, found, _children.length - found - 1 );
                System.arraycopy( _shapeRelations, found + 1, newShapeRelations, found, _children.length - found - 1 );
                _children = newChildren;
                _shapeRelations = newShapeRelations;
            }
        }

    }

    public String toString() {
        return "Relation[parent=" + _parent + ",children=" + java.util.Arrays.deepToString(_children) + ",shapeRelations=" + java.util.Arrays.deepToString(_shapeRelations) + "]";
    }

    public Optional<Tensor<V>> getParent() {
        return Optional.ofNullable( _parent );
    }
}
