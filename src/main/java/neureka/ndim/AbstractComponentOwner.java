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
import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.devices.Device;
import neureka.devices.opencl.OpenCLDevice;
import neureka.framing.Relation;
import neureka.optimization.Optimizer;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
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
    private final static Map<Class<?>, Integer> _CLASS_ORDER = new HashMap<>();
    static {
            _CLASS_ORDER.put(Optimizer.class,	1   );
            _CLASS_ORDER.put(JITProp.class,	2   );
            _CLASS_ORDER.put(OpenCLDevice.class,	3   );
            _CLASS_ORDER.put(Tsr.class,	4  );
            _CLASS_ORDER.put(Relation.class,	5 );
            _CLASS_ORDER.put(Device.class,	6 );
            _CLASS_ORDER.put(GraphNode.class,	7 );
    }

    /**
     *  (Tensor) components
     *  A collection of component.
     */
    protected Component<InstanceType>[] _components = null;

    private synchronized void _mergeComps( Component<InstanceType>[] components ) {
        if ( _components == null ) _components = components;
        else if ( components != null ) {
            Component<InstanceType>[] newComponents = new Component[ _components.length + components.length ];
            System.arraycopy( _components, 0, newComponents, 0, _components.length );
            System.arraycopy( components, 0, newComponents, _components.length, components.length );
            _components = newComponents;
            for ( int i = 1; i < _components.length; i++ ) {
                Component<InstanceType> a = _components[ i-1 ];
                Component<InstanceType> b = _components[ i ];
                int orderA = ( a == null || !_CLASS_ORDER.containsKey(a.getClass()) ) ? 0 : _CLASS_ORDER.get( a.getClass() );
                int orderB = ( b == null || !_CLASS_ORDER.containsKey(b.getClass())) ? 0 : _CLASS_ORDER.get( b.getClass() );
                if ( orderB > orderA ) {
                    _components[ i - 1 ] = b;
                    _components[ i ] = a;
                }
            }
        }
    }

    private synchronized void _removeComp( Component<InstanceType> component ) {
        if ( _components != null && _components.length != 0 ) {
            int count = 0;
            for ( int i = 0; i < _components.length; i++ )
                if ( _components[ i ] == component ) _components[ i ] = null;
                else count++;
            assert count == _components.length -1;
            Component<InstanceType>[] newComponents = new Component[ count ];
            if ( count != _components.length ) {
                count = 0;
                for ( int i = 0; i < _components.length; i++ )
                    if ( _components[ i ] == null ) count++;
                    else newComponents[ i - count ] = _components[ i ];
                _components = newComponents;
            }
        }
    }

    protected void _transferFrom( AbstractComponentOwner<InstanceType> other ) {
        if ( other._components != null ) { // Inform components about their new owner:
            _mergeComps( other._components );
            for ( Component<InstanceType> o : other._components ) if (o!=null) o.update(
                    (InstanceType) other, (InstanceType) this
            );
        }
        other._delComps();
    }

    protected void _delComps() {
        _components = null;
    }

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
    public <T extends Component<?>> T find( Class<T> componentClass )
    {
        if ( _components != null) {
            for ( Component<?> component : _components ) {
                if ( componentClass.isInstance( component ) ) return (T) component;
            }
        }
        return null;
    }

    /**
     *  This method removes a component identified by the passed Class
     *  instance if found in the stored component collection.
     *
     * @param componentClass The type/class of a component which will be removed by this method.
     * @param <T> The type parameter of the component which will be removed by this method.
     * @return This very class.
     */
    public <T extends Component<InstanceType>> InstanceType remove( Class<T> componentClass )
    {
        T oldComponent = find( componentClass );
        if ( oldComponent != null ) {
            _removeComp( _removeOrReject( oldComponent ) );
            //_components.trimToSize();
        }
        if ( _components != null && _components.length == 0 ) {
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
    public <T extends Component<InstanceType>> boolean has( Class<T> componentClass ) {
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
                _removeComp( oldCompartment );
            }
        }
        Component<InstanceType> newComp = _setOrReject( newComponent );
        if ( newComp != null ) _mergeComps( new Component[]{newComp} );
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
     * into the provided Consumer lambda.
     * If however no component was found then this lambda is being left untouched.
     *
     * @param cc Component class of whose type the requested component is.
     * @param action An action applied on the requested component if found.
     * @return True if a component could be found, false otherwise.
     */
    public <T extends Component<InstanceType>> boolean forComponent( Class<T> cc, Consumer<T> action ) {
        T component = this.find( cc );
        if ( component != null ) {
            action.accept( component );
            return true;
        } else return false;
    }


}
