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

package neureka.common.composition;

import neureka.Tsr;
import neureka.autograd.GraphNode;
import neureka.backend.api.BackendContext;
import neureka.common.utility.LogUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

/**
 *  Together with the {@link Component} interface, this class defines a simple
 *  component system in which implementations of the {@link Component} interface
 *  are managed by extensions of this {@link AbstractComponentOwner}.
 *  An {@link AbstractComponentOwner} can have multiple {@link Component} instances which
 *  are being accessed via the {@link Class} objects of these implementations.
 *  This means that the {@link AbstractComponentOwner} can only reference a single
 *  instance of a concrete {@link Component} implementation class.                                   <br>
 *                                                                                                   <br>
 *  This class is also used as the root precursor class of the concrete {@link Tsr} class
 *  from which tensor instances can be created, but also the precursor class of
 *  the {@link BackendContext} which is managed by thread local
 *  {@link neureka.Neureka} library context instances.
 *  An {@link BackendContext} directly or indirectly
 *  hosts {@link neureka.backend.api.Operation}, {@link java.util.function.Function}
 *  and of course {@link Component} implementations.
 *  Tensors on the other hand use this component system to enable autograd
 *  via the {@link GraphNode} class and to reference gradients ({@link Tsr} is itself a {@link Component})
 *  among other type of components...
 *
 * @param <C> The concrete class at the bottom end of the inheritance hierarchy. (Used to allow for method chaining)
 */
public abstract class AbstractComponentOwner<C> implements ComponentOwner<C>
{
    /**
     *  An array of (type) unique components.
     */
    private Component<C>[] _components = null;


    protected C _this() { return (C) this; }

    private synchronized void _setComps( Component<C>[] components ) { _components = components; }

    private synchronized void _addOrRemoveComp( Component<C> component, boolean remove ) {
        boolean[] changeExecuted = { false };
        if ( remove ) {
            boolean removalAccepted =
                    component.update(
                        new Component.OwnerChangeRequest<C>() {
                            @Override public C getOldOwner() { return _this(); }
                            @Override public C getNewOwner() { return null; }
                            @Override public boolean executeChange() {
                                _remove( component );
                                changeExecuted[ 0 ] = true;
                                return true; // We inform the component that the change was executed successfully!
                            }
                        }
                    );
            if ( removalAccepted && !changeExecuted[0] ) _remove( component );
        } else {
            // The component receives an initial update call:
            if ( component != null ) { 
                boolean additionAccepted =
                        component.update(
                            new Component.OwnerChangeRequest<C>() {
                                @Override public C getOldOwner() { return null; }
                                @Override public C getNewOwner() { return _this(); }
                                @Override public boolean executeChange() {
                                    _add( _setOrReject( component ) );
                                    changeExecuted[ 0 ] = true;
                                    return true; // We inform the component that the change was executed successfully!
                                }
                            }
                        );
                if ( additionAccepted && !changeExecuted[0] ) _add( _setOrReject( component ) );
            }
        }
    }

    private void _remove( Component<C> component ) {
        LogUtil.nullArgCheck( component, "component", Component.class );
        if ( _components != null && _components.length != 0 )
            for ( int i = 0; i < _components.length; i++ )
                if ( _components[ i ] == component ) {
                    _setComps( _newArrayWithout(i, _components) );
                    break;
                }
    }

    private static <C> Component<C>[] _newArrayWithout(int index, Component<C>[] array) {
        Component<C>[] newArray = new Component[array.length - 1];
        if ( index >= 0 )
            System.arraycopy(array, 0, newArray, 0, index);
        if ( array.length - (index + 1) >= 0 )
            System.arraycopy(array, index + 1, newArray, index, array.length - (index + 1));
        return newArray;
    }

    private void _add( Component<C> component ) {
        LogUtil.nullArgCheck( component, "component", Component.class );
        if ( _components == null ) _setComps( new Component[]{ component } );
        else {
            for ( Component<C> c : _components ) if ( c == component ) return;
            Component<C>[] newComponents = new Component[ _components.length + 1 ];
            System.arraycopy( _components, 0, newComponents, 0, _components.length );
            newComponents[ newComponents.length - 1 ] = component;
            _setComps( newComponents );
        }
    }

    /**
     *  A component owner might need to exchange components. <br>
     *  Meaning that the components of another owner will be transferred and adopted by the current one.
     *  During this process the transferred components will be notified of their new owner.
     *  This is important because some components might reference their owners... <br>
     *  <br>
     *  This change happens for example in the {@link Tsr} class when tensors are being instantiated by
     *  certain constructors which require the injection of the contents of another tensor into a new one.
     *
     * @param other The other owner which will be stripped of its components which are then incorporated into this owner.
     */
    protected void _transferFrom( AbstractComponentOwner<C> other ) {
        if ( other._components != null ) {
            _setComps( other._components ); // Inform components about their new owner:
            for ( Component<C> c : _components )
                c.update(
                        new Component.OwnerChangeRequest<C>() {
                            @Override public C       getOldOwner()   { return other._this(); }
                            @Override public C       getNewOwner()   { return _this();       }
                            @Override public boolean executeChange() { return false;         }
                        }
                    );
            other._deleteComponents();
        }
    }

    /**
     *  This method deletes the array of components of this component owner
     *  by nulling the array variable field.
     */
    protected void _deleteComponents() { _components = null; }

    /**
     *  This method tries to find a component inside the internal
     *  component array whose class matches the one provided.
     *  If no such component could be found then
     *  the return value will simply be null.
     *
     * @param componentClass The type/class of the component which shall be found and returned.
     * @param <T> The type parameter defining the component class.
     * @return The correct component or null if nothing has been found.
     */
    @Override
    public <T extends Component<?>> T get(Class<T> componentClass)
    {
        LogUtil.nullArgCheck( componentClass, "componentClass", Class.class );
        if ( _components != null ) {
            for ( int i = 0; i < _components.length; i++ ) {
                if ( componentClass.isInstance( _components[ i ] ) ) {
                    Component<C> component = _components[ i ];
                    if ( _components.length > 1 && i > 0  ) {
                        // Now we swap the components (faster access for common components):
                        _components[ i ] = _components[ i - 1 ];
                        _components[ i - 1 ] = component;
                    }
                    return (T) component;
                }
            }
        }
        return null;
    }

    /**
     *  This method tries to find all components inside the internal
     *  component array whose classes are sub types of the one provided.
     *  If no such components could be found then
     *  the return value will simply be an empty list.
     *
     * @param componentClass The type/class of the components which shall be found and returned as list.
     * @param <T> The type parameter defining the component class.
     * @return The correct component or null if nothing has been found.
     */
    @Override
    public <T extends Component<?>> List<T> getAll(Class<T> componentClass) {
        LogUtil.nullArgCheck( componentClass, "componentClass", Class.class );
        List<T> found = new ArrayList<>();
        if ( _components != null ) {
            for ( Component<?> component : _components ) {
                if (
                    component != null &&
                    componentClass.isAssignableFrom( component.getClass() )
                )
                    found.add((T) component);
            }
        }
        return found;
    }

    /**
     *  This method removes a component identified by the passed Class
     *  instance if found in the stored component collection.
     *
     * @param componentClass The type/class of a component which will be removed by this method.
     * @param <T> The type parameter of the component which will be removed by this method.
     * @return This very class.
     */
    @Override
    public <T extends Component<C>> C remove(Class<T> componentClass)
    {
        LogUtil.nullArgCheck( componentClass, "componentClass", Class.class );
        T oldComponent = get( componentClass );
        if ( oldComponent != null ) _addOrRemoveComp( _removeOrReject( oldComponent ), true );
        if ( _components != null && _components.length == 0 ) _components = null;
        return _this();
    }

    /**
     * This method checks if a component identified by the passed {@link Class}
     * instance is present inside the stored component collection.
     *
     * @param componentClass The class/type of component that might exist in components.
     * @return True if the component of the given type/class has been found.
     */
    @Override
    public <T extends Component<C>> boolean has(Class<T> componentClass) {
        LogUtil.nullArgCheck( componentClass, "componentClass", Class.class );
        return get( componentClass ) != null;
    }

    /**
     * This methods stores the passed component inside the component
     * collection of this class...
     * However, it only adds the provided component if it is not
     * "rejected" by an abstract method, namely : "_addOrReject" !
     * Rejection means that this method simply returns null.
     *
     * @param newComponent The new component which should be added to the components list.
     * @return This very class.
     */
    @Override
    public <T extends Component<C>> C set(T newComponent)
    {
        LogUtil.nullArgCheck( newComponent, "newComponent", Component.class );
        Component<C> oldCompartment;
        if ( _components != null ) {
            oldCompartment = (Component<C>) get( newComponent.getClass() );
            if ( oldCompartment != null )
                _addOrRemoveComp( oldCompartment, true );
        }
        _addOrRemoveComp( newComponent, false );
        return _this();
    }

    protected <T> void _set( Component<T> anyComponent ) {
        this.set( (Component<C>) anyComponent);
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
    protected abstract <T extends Component<C>> T _setOrReject( T newComponent );

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
    protected abstract <T extends Component<C>> T _removeOrReject( T newComponent );

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
    @Override
    public <T extends Component<C>> boolean forComponent(Class<T> cc, Consumer<T> action) {
        T component = this.get( cc );
        if ( component != null ) {
            action.accept( component );
            return true;
        }
        else return false;
    }


}
