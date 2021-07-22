package neureka;

/**
 *  This interface defines the functionality of the component of a simple component system used by Neureka.
 *  Currently the only implemented component system using this interface is that of the Tsr class.
 *  Meaning that said class is the "owner" of components as is defined by the type parameter below.
 *
 * @param <OwnerType> The type of which an implementation of this interface is a component.
 */
public interface Component<OwnerType>
{
    /**
     *  This method informs the component about an owner switch.
     *  If components hold references of their owner then this
     *  method gives them the ability to update said reference
     *  when a new owner takes over the components of an old one.
     *
     * @param oldOwner The previous owner type instance.
     * @param newOwner The new owner type instance.
     */
    void update( OwnerType oldOwner, OwnerType newOwner );

}
