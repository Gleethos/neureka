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

    An early precursor class of the AbstractNDArray class and the Tsr class...

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

import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;

/**
 *  This is the root precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  {@link Tsr} inherits from {@link AbstractNDArray} which inherits from {@link AbstractComponentOwner}
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 * @param <InstanceType> The class at the bottom end of the inheritance hierarchy. (Used for factory pattern)
 */
public abstract class AbstractComponentOwner<InstanceType>
{
    /**
     *  The following static map enables fast access to properties which describe
     *  the "importance" of an implementation of the Component interface.
     *  This is relevant only for performance reasons because
     *  the component owner referencing this component among
     *  others can store them according to their order to
     *  make component access as fast as possible! <br>
     *  There is not much more to this then that.
     *  New component implementations will default to a class order of 0
     *  and otherwise one should consider profiling the access patterns
     *  of the component system and update this mapping...
     *
     */
    private static final Map<Class<? extends Component>, Integer> _CLASS_ORDER = new HashMap<>();
    static {
            _CLASS_ORDER.put(Optimizer.class,	    1   );
            _CLASS_ORDER.put(JITProp.class,	        2   );
            _CLASS_ORDER.put(OpenCLDevice.class,	3   );
            _CLASS_ORDER.put(Tsr.class,	            4   );
            _CLASS_ORDER.put(Relation.class,	    5   );
            _CLASS_ORDER.put(Device.class,	        6   );
            _CLASS_ORDER.put(GraphNode.class,	    7   );
    }

    /**
     *  A collection of components.
     */
    private Component<InstanceType>[] _components = null;

    private synchronized void _setComps( Component<InstanceType>[] components ) {
        _components = components;
    }

    private synchronized void _addOrRemoveComp( Component<InstanceType> component, boolean remove ) {
        if ( remove ) {
            if ( _components != null && _components.length != 0 && component != null ) {
                int count = 0;
                for ( int i = 0; i < _components.length; i++ )
                    if ( _components[ i ] == component ) _components[ i ] = null;
                    else count++;
                if ( count != _components.length ) {
                    Component<InstanceType>[] newComponents = new Component[ count ];
                    count = 0;
                    for ( int i = 0; i < _components.length; i++ )
                        if ( _components[ i ] == null ) count++;
                        else newComponents[ i - count ] = _components[ i ];
                    _components = newComponents;
                }
            }
        } else {
            if ( _components == null ) _setComps( new Component[]{ component } );
            else if ( component != null ) {
                for ( Component<InstanceType> c : _components ) if ( c == component ) return;
                Component<InstanceType>[] newComponents = new Component[ _components.length + 1 ];
                System.arraycopy( _components, 0, newComponents, 0, _components.length );
                newComponents[ newComponents.length - 1 ] = component;
                _setComps( newComponents );
                for ( int i = 1; i < _components.length; i++ ) {
                    Component<InstanceType> a = _components[ i-1 ];
                    Component<InstanceType> b = _components[ i ];
                    int orderA = _CLASS_ORDER.getOrDefault( a, 0 );
                    int orderB = _CLASS_ORDER.getOrDefault( b, 0 );
                    if ( orderB > orderA ) {
                        _components[ i - 1 ] = b;
                        _components[ i ] = a;
                    }
                }
            }
        }
    }

    /**
     *  A component owner might need to "changes its identity". <br>
     *  Meaning that the components of another owner will be stripped of its components
     *  which will be adopted by the current one.
     *  During this process the transferred components will be notified of their new owner.
     *  This is important because some components might reference their owners... <br>
     *  <br>
     *  This change currently only happens in the 'Tsr' sub-class when tensors are being instantiated by
     *  certain constructors to which input tensors and a math expression is passed.
     *  This triggers the creation of a Function instance and execution on the provided
     *  input tensors. In that case the output tensor will be created somewhere
     *  along the execution call stack, however the result is expected to be
     *  stored within the tensor whose constructor initialized all of this.
     *  In that case this tensor will rip out the guts of the resulting output
     *  tensor and stuff onto its own field variables.
     *
     * @param other The other owner which will be stripped of its components which are then incorporated into this owner.
     */
    protected void _transferFrom( AbstractComponentOwner<InstanceType> other ) {
            if ( other._components != null ) {
            _setComps( other._components ); // Inform components about their new owner:
            for ( Component<InstanceType> c : _components ) c.update((InstanceType) other, (InstanceType) this);
            other._deleteComponents();
        }
    }

    /**
     *  This method deletes the array of components of this component owner
     *  by nulling the array variable field.
     */
    protected void _deleteComponents() {
        _components = null;
    }

    /**
     *  This method tries to find a component inside the stored
     *  component array whose class matches the one provided.
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
        if ( oldComponent != null ) _addOrRemoveComp( _removeOrReject( oldComponent ), true );
        if ( _components != null && _components.length == 0 ) _components = null;
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
        Component<InstanceType> oldCompartment;
        if ( _components != null ) {
            oldCompartment = (Component<InstanceType>) find( newComponent.getClass() );
            if ( oldCompartment != null ) {
                _addOrRemoveComp( oldCompartment, true );
            }
        }
        _addOrRemoveComp( _setOrReject( newComponent ), false );
        return (InstanceType) this;
    }

    /**
     * This abstract method ought to be implemented further down
     * the inheritance hierarchy where it's responsibility
     * makes more sense, namely :
     * An implementation of this method checks if the passed component
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
     * This method abstract ought to be implemented further down
     * the inheritance hierarchy where it's responsibility
     * makes more sense, namely :
     * An implementation of this method checks if the passed component
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
