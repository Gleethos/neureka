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

import java.util.Collection;
import java.util.Iterator;
import java.util.Spliterator;
import java.util.function.IntFunction;

public abstract class AbstractBaseDevice<ValType> implements Device<ValType>, Component<Tsr<ValType>>
{

    @Override
    public int size() {
        Collection<Tsr<ValType>> tensors = this.getTensors();
        if ( tensors == null ) return 0;
        return tensors.size();
    }

    @Override
    public boolean isEmpty() {
        return this.size() == 0;
    }

    @Override
    public boolean contains( Tsr<ValType> o ) {
        return this.getTensors().contains( o );
    }

    @Override
    public Iterator<Tsr<ValType>> iterator() {
        return this.getTensors().iterator();
    }

    @Override
    public <T> T[] toArray( IntFunction<T[]> generator ) {
        return null; // this.getTensors().toArray( generator );
    }

    @Override
    public Spliterator<Tsr<ValType>> spliterator() {
        return getTensors().spliterator();
    }

}
