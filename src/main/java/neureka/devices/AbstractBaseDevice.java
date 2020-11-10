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

           _         _                  _   ____                 _____             _
     /\   | |       | |                | | |  _ \               |  __ \           (_)
    /  \  | |__  ___| |_ _ __ __ _  ___| |_| |_) | __ _ ___  ___| |  | | _____   ___  ___ ___
   / /\ \ | '_ \/ __| __| '__/ _` |/ __| __|  _ < / _` / __|/ _ \ |  | |/ _ \ \ / / |/ __/ _ \
  / ____ \| |_) \__ \ |_| | | (_| | (__| |_| |_) | (_| \__ \  __/ |__| |  __/\ V /| | (_|  __/
 /_/    \_\_.__/|___/\__|_|  \__,_|\___|\__|____/ \__,_|___/\___|_____/ \___| \_/ |_|\___\___|


*/

package neureka.devices;

import neureka.Component;
import neureka.Tsr;
import org.jetbrains.annotations.NotNull;

import java.util.Collection;
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
