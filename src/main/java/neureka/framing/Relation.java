package neureka.framing;

import neureka.Component;
import neureka.Tsr;

import java.lang.ref.WeakReference;
import java.util.function.Consumer;

public class Relation implements Component<Tsr> {

    private Tsr _parent;// Children need their parents. They shall not be garbage collected.

    private WeakReference<Tsr>[] _children;// Children may be garbage collected if not needed.

    @Override
    public void update(Tsr oldOwner, Tsr newOwner){
        if (_parent != null) {
            Relation pr = _parent.find(Relation.class);
            for (int i=0; i < pr._children.length; i++) {
                if (pr._children[i].get() == oldOwner) {
                    pr._children[i] = new WeakReference<>(newOwner);
                }
            }
        }
        if (_children != null) {
            for (WeakReference<Tsr> c : _children) {
                Tsr t = c.get();
                if ( t != null ) {
                    Relation cr = t.find(Relation.class);
                    if ( cr != null ) cr._parent = newOwner;
                }
            }
        }
    }


    public Relation addParent(Tsr parent){
        _parent = parent;
        return this;
    }


    public Relation addChild(Tsr child){
        if ( _children == null ) {
            _children = new WeakReference[]{new WeakReference(child)};
        } else {
            WeakReference<Tsr>[] newChildren = new WeakReference[_children.length+1];
            System.arraycopy(_children, 0, newChildren, 0, _children.length);
            newChildren[_children.length] = new WeakReference(child);
            _children = newChildren;
        }
        return this;
    }


    public Relation foreachChild(Consumer<Tsr> action){
        if ( _children != null ) {
            for ( WeakReference<Tsr> r : _children ) {
                Tsr c = r.get();
                if ( c != null ) {
                    action.accept(c);
                    Relation relation = (Relation) c.find(Relation.class);
                    if ( relation != null ) relation.foreachChild(action);
                }
            }
        }
        return this;
    }

    public Tsr findRootTensor() {
        if ( _parent==null ) return null;
        else if ( !_parent.has(Relation.class) ) return null;
        else if ( !_parent.find(Relation.class).hasParent() ) return _parent;
        else {
            return _parent.find(Relation.class).findRootTensor();
        }
    }

    public boolean hasParent(){
        return _parent!=null;
    }

    public boolean hasChildren(){
        return _children!=null;
    }

    public int childCount(){
        return (_children==null) ? 0 : _children.length;
    }

}
