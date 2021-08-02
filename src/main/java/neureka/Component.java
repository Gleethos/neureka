package neureka;

/**
 *  This interface alongside the {@link neureka.ndim.AbstractComponentOwner} class define a simple component system.
 *  The component type defined by this interface is used to create components for the {@link Tsr} class
 *  as well as the {@link neureka.backend.api.OperationContext} class which both directly or indirectly
 *  extend the {@link neureka.ndim.AbstractComponentOwner} class.
 *  The type parameter of this interface represents the "owner" of the {@link Component}.
 *
 * @param <O> The owner type of which an implementation of this interface is a component.
 */
public interface Component<O>
{
    /**
     *  Entries of this enum represent events describing updates to the state
     *  of the owner of a given {@link Component} instance.
     */
    enum IsBeing { REMOVED, ADDED, REPLACED, UPDATED }

    /**
     *  {@link OwnerChangeRequest} implementation instances will be passed to
     *  the {@link Component#update(OwnerChangeRequest)} method which inform a
     *  given component about a state change related to said component.
     *  They are used by component owners to communicate and
     *  negotiate update events to their components using the {@link IsBeing} enum and
     *  some useful methods providing both a context for a component as well as the ability
     *  for the component to trigger the state change itself.
     *
     * @param <O> The type parameter representing the concrete type of the component owner.
     */
    interface OwnerChangeRequest<O>
    {
        /**
         * @return The previous owner type instance or null if the component is being added to the owner.
         */
        O getOldOwner();

        /**
         * @return The new owner type instance.
         */
        O getNewOwner();

        /**
         *  This method will trigger the actual state change identified by the {@link IsBeing}
         *  constant returned by the {@link #type()} method.
         *  It exists so that a component can decide when the change should occur.
         *  If the change type is set to {@link IsBeing#ADDED} for example then this would
         *  mean that after calling this method, the current component will now be a
         *  component of the current component owner.
         *
         * @return The truth value determining if the state change was successfully executed.
         */
        boolean executeChange();

        /**
         *  This method will return one of the following states:
         *  {@link IsBeing#ADDED}, {@link IsBeing#REMOVED}, {@link IsBeing#REPLACED}, {@link IsBeing#UPDATED}
         *
         * @return The type of change that is about to happen to the component receiving this.
         */
        default IsBeing type() {
            if ( getOldOwner() != null && getNewOwner() != null ) return IsBeing.REPLACED;
            if ( getOldOwner() != null && getNewOwner() == null ) return IsBeing.REMOVED;
            if ( getOldOwner() == null && getNewOwner() != null ) return IsBeing.ADDED;
            return IsBeing.UPDATED;
        }
    }

    /**
     *  Components are not the slaves of their owners.
     *  If the owner registers any state changes related to a given component, then
     *  said component will be informed by the owner about the change as well as receive
     *  the ability to decide when the change should occur or if the change should occur at all.
     *  This method informs the component about state changes within the owner
     *  A typical state change would be an owner switch or simply that this component
     *  is being added to, or removed from, its current owner.
     *  If components hold references to their owners then this method gives them
     *  the ability to update said reference when a new owner takes over the components of an old one.
     *  The {@link OwnerChangeRequest} implementation instance passed to this method
     *  informs this component about the current state change and its type ({@link OwnerChangeRequest#type()}).
     *  If this method returns false then this means that this component rejects the proposed update.
     *  The component owner will then abort the proposed change.
     *
     * @param changeRequest An {@link OwnerChangeRequest} implementation instance used to communicate the type of change, context information and the ability to execute the change directly.
     * @return The truth value determining if the state change should be aborted or not.
     */
    boolean update( OwnerChangeRequest<O> changeRequest );

}
