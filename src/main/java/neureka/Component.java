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

    enum IsBeing { REMOVED, ADDED, REPLACED, UPDATED }
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

        boolean executeChange();

        default IsBeing type() {
            if ( getOldOwner() != null && getNewOwner() != null ) return IsBeing.REPLACED;
            if ( getOldOwner() != null && getNewOwner() == null ) return IsBeing.REMOVED;
            if ( getOldOwner() == null && getNewOwner() != null ) return IsBeing.ADDED;
            return IsBeing.UPDATED;
        }
    }

    /**
     *  This method informs the component about an owner switch.
     *  If components hold references of their owner then this
     *  method gives them the ability to update said reference
     *  when a new owner takes over the components of an old one.
     *  The method is also called when the component is initially
     *  added to the owner, in which case the "oldOwner"
     *  is going to be null.
     */
    void update( OwnerChangeRequest<O> changeRequest );

}
