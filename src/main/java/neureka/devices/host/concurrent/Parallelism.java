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
*/
package neureka.devices.host.concurrent;

import neureka.devices.host.machine.ConcreteMachine;

import java.util.function.IntSupplier;

/**
 * A set of standard levels of parallelism derived from the number of available cores and optionally capped by
 * reserving a specified amount of memory per thread. The info about available cores/threads/memory comes from
 * {@link ConcreteMachine#ENVIRONMENT}.
 */
public enum Parallelism implements IntSupplier
{
    /**
     * The total number of threads (incl. hyper-threads)
     */
    THREADS(() -> ConcreteMachine.ENVIRONMENT.threads),
    /**
     * The number of CPU cores
     */
    CORES(() -> ConcreteMachine.ENVIRONMENT.cores),
    /**
     * The number of top level (L2 or L3) cache units
     */
    UNITS(() -> ConcreteMachine.ENVIRONMENT.units),
    /**
     * 8
     */
    EIGHT(() -> 8),
    /**
     * 4
     */
    FOUR(() -> 4),
    /**
     * 2
     */
    TWO(() -> 2),
    /**
     * 1
     */
    ONE(() -> 1);

    private final IntSupplier _supplier;


    Parallelism( final IntSupplier value ) { _supplier = value; }

    @Override
    public int getAsInt() { return _supplier.getAsInt(); }

}
