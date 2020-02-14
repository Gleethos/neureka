package neureka;

public interface Component<OwnerType>
{
    void update(OwnerType oldOwner, OwnerType newOwner);
}
