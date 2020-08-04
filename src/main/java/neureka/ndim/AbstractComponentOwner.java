package neureka.ndim;

import neureka.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Consumer;

/**
 *  This is the root precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  Tsr inherits from AbstractNDArray which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 */
public abstract class AbstractComponentOwner<InstanceType> {

    /**
     *  (Tensor) components
     */
    protected List<Component<InstanceType>> _components = Collections.synchronizedList(new ArrayList<>());

    /**
     * @param componentClass The type/class of the component which shall be found and returned.
     * @return The correct component or null if nothing has been found.
     */
    public <T> T find(Class<T> componentClass) {
        if (_components != null) {
            for (Component<InstanceType> component : _components) {
                if (componentClass.isInstance(component)) return (T)component;
            }
        }
        return null;
    }

    /**
     *
     * @param componentClass The type/class of a component which will be removed by this method.
     * @return This very class.
     */
    public <T extends Component<InstanceType>>  InstanceType remove(Class<T> componentClass) {
        T oldComponent = find(componentClass);
        if (oldComponent != null) {
            _components.remove(_removeOrReject(oldComponent));
            //_components.trimToSize();
        }
        if (_components != null && _components.size() == 0) {
            _components = null;
        }
        return (InstanceType)this;
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
    public <T extends Component<InstanceType>> InstanceType add(T newComponent)
    {
        if (newComponent == null) return (InstanceType)this;
        T oldCompartment = null;
        if (_components != null) {
            oldCompartment = (T) find(newComponent.getClass());
            if (oldCompartment != null) {
                _components.remove(oldCompartment);
                //_components.trimToSize();
            }
        } else _components = new ArrayList<>();

        _components.add(_addOrReject(newComponent));
        return (InstanceType)this;
    }

    /**
     * @param newComponent The component which should be added to the components list.
     * @return The same component or null if it has been rejected.
     */
    protected abstract <T extends Component<InstanceType>> T _addOrReject(T newComponent);


    /**
     * @param newComponent The component which should be removed from the components list.
     * @return The same component or null if its removal has been rejected.
     */
    protected abstract <T extends Component<InstanceType>> T _removeOrReject(T newComponent);

    /**
     *
     * @param cc Component class of whose type the requested component is.
     * @param action An action applied on the requested component if found.
     * @return True if a component could be found, false otherwise.
     */
    public <T extends Component<InstanceType>> boolean forComponent(Class<T> cc, Consumer<T> action) {
        T component = this.find(cc);
        if (component!=null) {
            action.accept(component);
            return true;
        } else return false;
    }


}
