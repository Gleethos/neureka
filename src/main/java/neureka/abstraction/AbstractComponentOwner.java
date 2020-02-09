package neureka.abstraction;

import java.util.ArrayList;
import java.util.function.Consumer;

/**
 *  This is the root precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  Tsr inherits from AbstractTensor which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 */
public abstract class AbstractComponentOwner {

    /**
     *  (Tensor) components
     */
    protected ArrayList<Object> _components = new ArrayList<Object>();

    /**
     * @param componentClass The type/class of the component which shall be found and returned.
     * @return The correct component or null if nothing has been found.
     */
    public Object find(Class componentClass) {
        if (_components != null) {
            for (int Pi = 0; Pi < _components.size(); Pi++) {
                if (componentClass.isInstance(_components.get(Pi))) {
                    return _components.get(Pi);
                }
            }
        }
        return null;
    }

    /**
     *
     * @param componentClass The type/class of a component which will be removed by this method.
     * @return This very class.
     */
    public AbstractComponentOwner remove(Class componentClass) {
        Object oldComponent = find(componentClass);
        if (oldComponent != null) {
            _components.remove(oldComponent);
            _components.trimToSize();
        }
        if (_components!=null && _components.size() == 0) {
            _components = null;
        }
        return this;
    }

    /**
     *
     * @param componentClass The class/type of a component that might exist in components.
     * @return True if the component of the given type/class has been found.
     */
    public boolean has(Class componentClass) {
        return find(componentClass) != null;
    }

    /**
     *
     * @param newComponent The new component which should be added to the components list.
     * @return This very class.
     */
    public AbstractComponentOwner add(Object newComponent) {
        if (newComponent == null) return this;
        Object oldCompartment = null;
        if (_components != null) {
            oldCompartment = find(newComponent.getClass());
            if (oldCompartment != null) {
                _components.remove(oldCompartment);
                _components.trimToSize();
            }
        } else {
            _components = new ArrayList<>();
        }
        _components.add(_addOrReject(newComponent));
        return this;
    }

    /**
     * @param newComponent The component which should be added to the components list.
     * @return The same component or null if it has been rejected.
     */
    protected abstract Object _addOrReject(Object newComponent);

    /**
     *
     * @param cc Component class of whose type the requested component is.
     * @param action An action applied on the requested component if found.
     * @return True if a component could be found, false otherwise.
     */
    public boolean forComponent(Class cc, Consumer<Object> action){
        Object component = this.find(cc);
        if(component!=null){
            action.accept(component);
            return true;
        } else {
            return false;
        }
    }


}
