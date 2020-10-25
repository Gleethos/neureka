package neureka.framing;

import neureka.Component;
import neureka.Tsr;

import java.lang.ref.WeakReference;
import java.util.function.Consumer;

public class Relation<ValueType> implements Component<Tsr<ValueType>> {

    private Tsr<ValueType> _parent;// Children need their parents. They shall not be garbage collected.

    private WeakReference<Tsr<ValueType>>[] _children;// Children may be garbage collected if not needed.

    @Override
    public void update( Tsr<ValueType> oldOwner, Tsr<ValueType> newOwner ) {
        if (_parent != null) {
            Relation<ValueType> pr = _parent.find(Relation.class);
            for (int i=0; i < pr._children.length; i++) {
                if (pr._children[ i ].get() == oldOwner) {
                    pr._children[ i ] = new WeakReference<>(newOwner);
                }
            }
        }
        if ( _children != null ) {
            for ( WeakReference<Tsr<ValueType>> c : _children ) {
                Tsr t = c.get();
                if ( t != null ) {
                    Relation<ValueType> cr = (Relation<ValueType>) t.find( Relation.class );
                    if ( cr != null ) cr._parent = newOwner;
                }
            }
        }
    }


    public Relation addParent( Tsr parent ) {
        _parent = parent;
        return this;
    }


    public Relation<ValueType> addChild( Tsr<ValueType> child ) {
        if ( _children == null ) {
            _children = new WeakReference[]{ new WeakReference( child ) };
        } else {
            WeakReference<Tsr<ValueType>>[] newChildren = new WeakReference[_children.length+1];
            System.arraycopy( _children, 0, newChildren, 0, _children.length );
            newChildren[_children.length] = new WeakReference(child);
            _children = newChildren;
        }
        return this;
    }


    public Relation<ValueType> foreachChild( Consumer<Tsr<ValueType>> action ) {
        if ( _children != null ) {
            for ( WeakReference<Tsr<ValueType>> r : _children ) {
                Tsr c = r.get();
                if ( c != null ) {
                    action.accept( c );
                    Relation relation = (Relation<ValueType>) c.find( Relation.class );
                    if ( relation != null ) relation.foreachChild( action );
                }
            }
        }
        return this;
    }

    public Tsr findRootTensor() {
        if ( _parent == null ) return null;
        else if ( !_parent.has( Relation.class ) ) return null;
        else if ( !_parent.find( Relation.class ).hasParent() ) return _parent;
        else {
            return _parent.find( Relation.class ).findRootTensor();
        }
    }

    public boolean hasParent(){
        return _parent != null;
    }

    public boolean hasChildren(){
        return _children != null;
    }

    public int childCount(){
        return ( _children == null ) ? 0 : _children.length;
    }

}
