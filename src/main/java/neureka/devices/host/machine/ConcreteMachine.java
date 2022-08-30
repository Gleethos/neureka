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

package neureka.devices.host.machine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class ConcreteMachine extends CommonMachine {

    private static final Logger _LOG = LoggerFactory.getLogger(ConcreteMachine.class);
    public static ConcreteMachine ENVIRONMENT = null;

    static {
        String architecture = ConcreteMachine.getArchitecture();
        long memory = ConcreteMachine.getMemory();
        int threads = ConcreteMachine.getThreads();

        for (Hardware hw : Hardware.PREDEFINED) {
            if (hw.architecture.equals(architecture) && (hw.threads == threads) && (hw.memory >= memory))
                ENVIRONMENT = hw.virtualize();
        }

        if (ENVIRONMENT == null) {
            _LOG.debug(
                "No matching hardware profile found for this system. " +
                "Instantiating a default hardware profile with the following main properties: " +
                "Architecture={} Threads={} Memory={}", architecture, threads, memory
            );
            ENVIRONMENT = Hardware.makeSimple(architecture, memory, threads).virtualize();
        }
    }

    private static final String AMD64 = "amd64";
    private static final String I386 = "i386";
    private static final String X86 = "x86";
    private static final String X86_64 = "x86_64";

    public static String getArchitecture() {

        // http://fantom.org/sidewalk/topic/756

        final String tmpProperty = System.getProperty("os.arch").toLowerCase();

        if (tmpProperty.equals(I386)) {
            return X86;
        } else if (tmpProperty.equals(AMD64)) {
            return X86_64;
        } else {
            return tmpProperty;
        }
    }

    public static long getMemory() {
        return Runtime.getRuntime().maxMemory();
    }

    public static int getThreads() {
        return Runtime.getRuntime().availableProcessors();
    }

    private final Hardware myHardware;
    private final Runtime myRuntime;

    ConcreteMachine(final Hardware hardware, final Runtime runtime) {

        super(hardware, runtime);

        myHardware = hardware;
        myRuntime = runtime;
    }

    @Override
    public boolean equals(final Object obj) {
        if ( this == obj ) return true;
        if ( !super.equals(obj) ) return false;
        if ( !(obj instanceof ConcreteMachine) ) return false;
        final ConcreteMachine other = (ConcreteMachine) obj;
        if ( myHardware == null )
            return other.myHardware == null;
        else
            return myHardware.equals(other.myHardware);
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = (prime * result) + ((myHardware == null) ? 0 : myHardware.hashCode());
        return result;
    }

    @Override
    public String toString() {
        return super.toString() + ((char)32) + myHardware.toString();
    }

}
