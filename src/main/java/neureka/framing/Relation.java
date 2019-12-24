package neureka.framing;

import neureka.Tsr;

import java.lang.ref.WeakReference;
import java.util.function.Consumer;

public class Relation {

    private WeakReference<Tsr>[] _children;
    private Tsr _parent;

    public Relation addParent(Tsr parent){
        _parent = parent;
        return this;
    }

    public Relation addChild(Tsr child){
        if(_children==null){
            _children = new WeakReference[]{new WeakReference(child)};
        } else {
            WeakReference<Tsr>[] newChildren = new WeakReference[_children.length+1];
            for(int i=0; i<_children.length; i++){
                newChildren[i] = _children[i];
            }
            newChildren[_children.length] = new WeakReference(child);
            _children = newChildren;
        }
        return this;
    }

    public Relation foreachChild(Consumer<Tsr> action){
        if(_children!=null){
            for(WeakReference<Tsr> r : _children){
                Tsr c = r.get();
                if(c!=null){
                    action.accept(c);
                    Relation parent = (Relation) c.find(Relation.class);
                    if(parent!=null) parent.foreachChild(action);
                }
            }
        }
        return this;
    }

    public boolean hasParent(){
        return _parent!=null;
    }

    public boolean hasChildren(){
        return _children!=null;
    }

    public int childCount(){
        return (_children==null)?0:_children.length;
    }

}
