package neureka.devices;

import neureka.Component;
import neureka.Tsr;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.IntFunction;
import java.util.function.Predicate;
import java.util.stream.Stream;

public abstract class AbstractBaseDevice<ValueType> implements Device<ValueType>, Component<Tsr<ValueType>>
{

    @Override
    public int size() {
        Collection<Tsr<ValueType>> tensors = this.getTensors();
        if ( tensors == null ) return 0;
        return tensors.size();
    }

    @Override
    public boolean isEmpty() {
        return this.size() == 0;
    }

    @Override
    public boolean contains(Object o) {
        return this.getTensors().contains( o );
    }

    @NotNull
    @Override
    public Iterator<Tsr<ValueType>> iterator() {
        return this.getTensors().iterator();
    }

    @NotNull
    @Override
    public Object[] toArray() {
        return this.getTensors().toArray();
    }

    @NotNull
    @Override
    public <T> T[] toArray(@NotNull T[] a) {
        return this.getTensors().toArray(a);
    }

    @Override
    public <T> T[] toArray(IntFunction<T[]> generator) {
        return this.getTensors().toArray(generator);
    }

    @Override
    public boolean add(Tsr<ValueType> tsr) {
        try {
            this.store( tsr );
        } catch ( Exception e ) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    @Override
    public boolean remove( Object o ) {
        if ( !(o instanceof Tsr) ) return false;
        try {
            this.free( (Tsr) o );
        } catch ( Exception e ) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    @Override
    public boolean containsAll(@NotNull Collection<?> c) {
        return this.getTensors().containsAll( c );
    }

    @Override
    public boolean addAll(@NotNull Collection<? extends Tsr<ValueType>> c) {
        for ( Tsr<ValueType> t : c ) {
            if ( !this.add( t ) ) return false;
        }
        return true;
    }

    @Override
    public boolean removeAll(@NotNull Collection<?> c) {
        return false;
    }

    @Override
    public boolean removeIf( Predicate<? super Tsr<ValueType>> filter ) {
        int removed = 0;
        for ( Tsr t : this.getTensors() ) removed += ( (this.remove(t))?1:0 );
        return removed > 0;
    }

    @Override
    public boolean retainAll(@NotNull Collection<?> c) {
        return false;
    }

    @Override
    public void clear() {
        this.dispose();
    }

    @Override
    public Spliterator<Tsr<ValueType>> spliterator() {
        return getTensors().spliterator();
    }

    @Override
    public Stream<Tsr<ValueType>> stream() {
        return this.getTensors().stream();
    }

    @Override
    public Stream<Tsr<ValueType>> parallelStream() {
        return this.getTensors().stream();
    }
}
