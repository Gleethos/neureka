package neureka.common.composition;

import java.util.List;
import java.util.function.Consumer;

/**
 *  A component owner is a thing holding components which can be accessed by their type class.
 *  This interface is used to create extensible APIs through flexible composition.
 *
 * @param <C> The concrete type of the component implementing this interface.
 */
public interface ComponentOwner<C>
{
    /**
     * Use this to get the component of the specified component type class.
     * @param componentClass The class of the component to be retrieved.
     * @return The component of the specified type class.
     * @param <T> The type of the component to be retrieved.
     */
    <T extends Component<?>> T get( Class<T> componentClass );

    /**
     * Use this to get all components of the specified component type class.
     * @param componentClass The class of the components to be retrieved.
     * @return A list of all components of the specified type class.
     * @param <T> The type of the components to be retrieved.
     */
    <T extends Component<?>> List<T> getAll( Class<T> componentClass );

    /**
     * Use this to remove a component of the specified component type class.
     * @param componentClass The class of the component to be removed.
     * @return This component owner instance (to allow for method chaining if so desired).
     * @param <T> The type of the component to be removed.
     */
    <T extends Component<C>> C remove( Class<T> componentClass );

    /**
     * Use this to check if a component of the specified component type class is present.
     * @param componentClass The class of the component to be checked.
     * @return True if a component of the specified type class is present, false otherwise.
     * @param <T> The type of the component to be checked.
     */
    <T extends Component<C>> boolean has( Class<T> componentClass );

    /**
     * Use this to set a component.
     * @param newComponent The new component to be set.
     * @return This component owner instance (to allow for method chaining if so desired).
     * @param <T> The type of the component to be set.
     */
    <T extends Component<C>> C set( T newComponent );

    /**
     * Use this to perform an action on a component of the specified component type class.
     * If no component of the specified type class is present, the provided consumer lambda
     * will not be invoked.
     * @param componentClass The class of the component to be acted upon.
     * @param action The action to be performed on the component.
     * @return True if a component of the specified type class was found and the action was performed, false otherwise.
     * @param <T> The type of the component to be acted upon.
     */
    <T extends Component<C>> boolean forComponent( Class<T> componentClass, Consumer<T> action );

}
