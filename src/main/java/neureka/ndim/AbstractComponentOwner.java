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

          _         _                  _    _____                                            _    ____
    /\   | |       | |                | |  / ____|                                          | |  / __ \
   /  \  | |__  ___| |_ _ __ __ _  ___| |_| |    ___  _ __ ___  _ __   ___  _ __   ___ _ __ | |_| |  |_|      ___ __   ___ _ __
  / /\ \ | '_ \/ __| __| '__/ _` |/ __| __| |   / _ \| '_ ` _ \| '_ \ / _ \| '_ \ / _ \ '_ \| __| |  |\ \ /\ / / '_ \ / _ \ '__|
 / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |__| (_) | | | | | | |_) | (_) | | | |  __/ | | | |_| |__| \ V  V /| | | |  __/ |
/_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|\____\___/|_| |_| |_| .__/ \___/|_| |_|\___|_| |_|\__|\____/ \_/\_/ |_| |_|\___|_|
                                                               | |
                                                               |_|

*/

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
 * @param <InstanceType> The class at the bottom end of the inheritance hierarchy. (Used for factory pattern)
 */
public abstract class AbstractComponentOwner<InstanceType>
{
    /**
     *  (Tensor) components
     *  A collection of component.
     */
    protected List<Component<InstanceType>> _components = Collections.synchronizedList(new ArrayList<>());

    /**
     *  This method tries to find a component inside the stored
     *  component collection whose class matches the one provided.
     *  If no such component could be found then
     *  the return value will simply be null.
     *
     * @param componentClass The type/class of the component which shall be found and returned.
     * @param <T> The type parameter defining the component class.
     * @return The correct component or null if nothing has been found.
     */
    public <T> T find(Class<T> componentClass)
    {
        if (_components != null) {
            for (Component<InstanceType> component : _components) {
                if (componentClass.isInstance(component)) return (T)component;
            }
        }
        return null;
    }

    /**
     *  This method removes a component identified by the passed Class
     *  instance if found in the stored component collection.
     *
     * @param componentClass The type/class of a component which will be removed by this method.
     * @return This very class.
     */
    public <T extends Component<InstanceType>> InstanceType remove(Class<T> componentClass)
    {
        T oldComponent = find( componentClass );
        if ( oldComponent != null ) {
            _components.remove( _removeOrReject( oldComponent ) );
            //_components.trimToSize();
        }
        if ( _components != null && _components.size() == 0 ) {
            _components = null;
        }
        return (InstanceType) this;
    }

    /**
     * This method checks if a component identified by the passed Class
     * instance is present inside the stored component collection.
     *
     * @param componentClass The class/type of a component that might exist in components.
     * @return True if the component of the given type/class has been found.
     */
    public boolean has( Class<?> componentClass ) {
        return find( componentClass ) != null;
    }

    /**
     * This methods stores the passed component inside the component
     * collection of this class...
     * However it only adds the provided component if it is not
     * "rejected" by an abstract method, namely : "_addOrReject" !
     * Rejection means that this method simply returns null.
     *
     * @param newComponent The new component which should be added to the components list.
     * @return This very class.
     */
    public InstanceType set( Component<InstanceType> newComponent)
    {
        if ( newComponent == null ) return (InstanceType) this;
        Component<InstanceType> oldCompartment = null;
        if ( _components != null ) {
            oldCompartment = (Component<InstanceType>) find( newComponent.getClass() );
            if ( oldCompartment != null ) {
                _components.remove( oldCompartment );
                //_components.trimToSize();
            }
        } else _components = new ArrayList<>();

        _components.add( _setOrReject( newComponent ) );
        return (InstanceType) this;
    }

    /**
     * This method ought to be implemented further down
     * the inheritance hierarchy where it's responsibility
     * makes more sense, namely :
     * An implementation of this method checks if the passes component
     * should be added or "rejected" to the component collection
     * of this class.
     * Rejection in this case simply means that it returns null instead
     * of the passed component.
     *
     * @param newComponent The component which should be added to the components list.
     * @return The same component or null if it has been rejected.
     */
    protected abstract <T extends Component<InstanceType>> T _setOrReject( T newComponent );


    /**
     * This method ought to be implemented further down
     * the inheritance hierarchy where it's responsibility
     * makes more sense, namely :
     * An implementation of this method checks if the passes component
     * should be removed from the component collection of this class
     * or its removal should be "rejected".
     * Rejection in this case simply means that it returns null instead
     * of the passed component.
     *
     * @param newComponent The component which should be removed from the components list.
     * @return The same component or null if its removal has been rejected.
     */
    protected abstract <T extends Component<InstanceType>> T _removeOrReject(T newComponent);

    /**
     * This method tries to find a stored component by identifying it
     * via the given Class instance in order to pass it
     * into th provided Consumer lambda.
     * If however no component was found then this lambda is being left untouched.
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
