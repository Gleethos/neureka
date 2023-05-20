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

     _   _                     _
    | \ | |                   | |
    |  \| | ___ _   _ _ __ ___| | ____ _
    | . ` |/ _ \ | | | '__/ _ \ |/ / _` |
    | |\  |  __/ |_| | | |  __/   < (_| |
    |_| \_|\___|\__,_|_|  \___|_|\_\__,_|

    This is a central singleton class used to configure the Neureka library.

*/

package neureka;


import neureka.backend.api.BackendContext;
import neureka.backend.api.Operation;
import neureka.backend.cpu.CPUBackend;
import neureka.backend.ocl.CLBackend;
import neureka.common.utility.LogUtil;
import neureka.common.utility.SettingsLoader;
import neureka.devices.host.CPU;
import neureka.devices.opencl.utility.Messages;
import neureka.dtype.DataType;
import neureka.dtype.custom.F64;
import neureka.ndim.config.types.sliced.SlicedNDConfiguration;
import neureka.view.NDPrintSettings;
import org.slf4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.ServiceLoader;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 *    {@link Neureka} is the key access point for thread local / global library settings ( see{@link Settings})
 *    as well as execution contexts (see {@link BackendContext})
 *    and pre-instantiated {@link neureka.math.Function}s.
 *    {@link Neureka} exposes the execution context via the {@link #backend()} method,
 *    the library settings which govern the behaviour of various library components
 *    can be accessed via the {@link #settings()} method.
 *    Common functions can be accessed within a given {@link BackendContext} instance based on which they were built.
 *    If one wishes to modify the default library settings it is possible to do so by editing
 *    the "library_settings.groovy" DSL file.
 */
public final class Neureka
{
    private static final ThreadLocal<Neureka> _INSTANCES;
    private static final Logger _LOG = org.slf4j.LoggerFactory.getLogger(Neureka.class);

    /**
     *  The current semantic version of this library build.
     */
    private static String _VERSION = "0.21.0";

    /**
     *  The truth value determining if OpenCL is available or not.
     */
    private static final boolean _OPENCL_AVAILABLE;

    static
    {
        _INSTANCES = new ThreadLocal<>();
        _OPENCL_AVAILABLE = Utility.isPresent( "org.jocl.CL", () -> Messages.findTip().bootstrapTip() );
    }

    private final Settings _settings;
    private final Utility _utility;

    /**
     *  This is a lazy reference to the so called {@link BackendContext}
     *  which will instantiated and populated as soon as the {@link #backend()}
     *  method is being called for the first time.
     *  This context contains anything needed to perform operations
     *  on tensors on using different {@link neureka.math.Function}
     *  or {@link neureka.devices.Device} implementation instances.
     */
    private BackendContext _backend;


    private Neureka() {
        _settings = new Settings();
        _utility = new Utility();
    }

    public BackendContext backend() {
        if ( _backend == null ) {
            _backend = new BackendContext();
            // loading operations!
            ServiceLoader<Operation> serviceLoader = ServiceLoader.load( Operation.class );
            Iterator<Operation> operationIterator = serviceLoader.iterator();

            try {
                // Iterating and logging if load was successful or not:
                while ( operationIterator.hasNext() ) {
                    try {
                        Operation operation = operationIterator.next();
                        assert operation.getIdentifier() != null;
                        assert operation.getOperator() != null;
                        if ( operation.getIdentifier() == null ) _LOG.error(_illegalStateFor( "function" ) );
                        if ( operation.getOperator() == null ) _LOG.error(_illegalStateFor( "operator" ) );
                        _backend.addOperation(operation);
                        _LOG.debug( LogUtil.format("Operation: '{}' loaded!", operation.getIdentifier()) );
                    } catch ( Exception e ) {
                        _LOG.error("Failed to load operations!", e);
                    }
                }
            } catch ( Exception e ) {
                _LOG.error("Failed to load operations!", e);
            }

            _backend.set( new CPUBackend() ); // CPU (JVM) is always available!

            if ( _OPENCL_AVAILABLE )
                _backend.set( new CLBackend() ); // OpenCL is available if the jocl dependency can find OpenCL drivers!
            else
                _LOG.debug( Messages.clContextCreationFailed() );
        }
        return _backend;
    }

    private static String _illegalStateFor( String type ) {
        return LogUtil.format(
                "Unexpected '{}' state encountered:\n" +
                        "The operation '{}' String should not be null but was null!",
                type, Operation.class.getSimpleName()
        );
    }

    /**
     *  The {@link Neureka} class represents the configuration of this library.
     *  Instances of this configuration are stored local to every thread in order to make
     *  both the library settings and the execution context threadsafe!
     *  This method will return the {@link Neureka} instance which corresponds to the thread calling it.
     *
     * @return The thread local library configuration state called {@link Neureka}.
     */
    public static Neureka get() {
        if ( Thread.currentThread().getName().startsWith(CPU.THREAD_PREFIX) )
            throw new IllegalAccessError(
                "Thread pool thread named '"+Thread.currentThread().getName()+"' may not " +
                "access thread local library instance directly!" +
                "This is because this settings instance is not representative of the main thread library context."
            );
        Neureka n = _INSTANCES.get();
        if ( n == null ) {
            n = new Neureka();
            synchronized ( Neureka.class ) {
                set( n );
                n.reset(); // Initial reset must be synchronized because of dependency issues!
            }
        }
        return n;
    }

    /**
     *  {@link Neureka} is a thread local singleton.
     *  Therefore, this method will only set the provided {@link Neureka} instance
     *  for the thread which is calling this method.
     *  Other threads calling the {@link #get()} method to retrieve the instance
     *  will get their own instance...
     *  (This can theoretically be bypassed by sharing instances)
     *
     * @param instance The {@link Neureka} instance which ought to be set as thread local singleton.
     */
    public static void set( Neureka instance ) {
        if ( Thread.currentThread().getName().startsWith(CPU.THREAD_PREFIX) )
            throw new IllegalAccessError(
                    "Thread pool thread named '"+Thread.currentThread().getName()+"' may not " +
                       "access thread local library instance directly! \n" +
                       "This is because this settings instance is not representative of the main thread library context."
                    );
        _INSTANCES.set(instance);
    }

    /**
     *  This allows you to configure Neureka using a Groovy DSL.
     *
     * @param closure A Groovy closure to allow for DSL type configuring.
     * @return The thread-local {@link Neureka} singleton instance.
     */
    public static Neureka configure( Object closure ) {
        Object o = SettingsLoader.tryGroovyClosureOn(closure, Neureka.get());
        if ( o instanceof String ) _VERSION = (String) o;
        return Neureka.get();
    }

    /**
     * @return The truth value determining if OpenCL is accessible.
     */
    public boolean canAccessOpenCL() {
        return _OPENCL_AVAILABLE &&
                get().backend()
                        .find(CLBackend.class)
                        .map( it -> it.getTotalNumberOfDevices() > 0 )
                        .orElse(false);
    }

    /**
     * @return The truth value determining if at least 1 {@link neureka.devices.opencl.OpenCLDevice} is accessible.
     */
    public boolean canAccessOpenCLDevice() {
        return canAccessOpenCL() &&
                get().backend()
                        .find(CLBackend.class)
                        .map( it -> it.getTotalNumberOfDevices() > 0 )
                        .orElse(false);
    }

    /**
     * @return An instance of library wide {@link Settings} determining the behaviour of many classes...
     */
    public Settings settings() { return _settings; }

    /**
     *  This allows you to configure Neureka using a Groovy DSL.
     */
    public Settings settings(Object closure) {
        SettingsLoader.tryGroovyClosureOn( closure, _settings );
        return _settings;
    }

    /**
     * @return An instance of an utility class useful for loading resources or checking if they are even available.
     */
    public Utility utility() { return _utility; }

    /**
     * @return The semantic version of the Neureka library.
     */
    public static String version() { return _VERSION; }

    /**
     *  This method will try to reload the "library_settings.groovy" script
     *  which will re-configure the library wide {@link Settings} instance nested inside {@link Neureka}.
     *  If the execution of this file fails then the settings will be reverted to a hardcoded default state.
     */
    public void reset() {
        try {
            SettingsLoader.loadProperties(this);
            // The following can be used when one desires a Groovy DSL as settings source!:
            //SettingsLoader.tryGroovyScriptsOn(this, script -> new GroovyShell(getClass().getClassLoader()).evaluate(script));
        } catch ( Exception e ) {
            settings().autograd().setIsRetainingPendingErrorForJITProp( true );
            settings().autograd().setIsApplyingGradientWhenTensorIsUsed( true );
            settings().autograd().setIsApplyingGradientWhenRequested( true );
            settings().debug().setIsKeepingDerivativeTargetPayloads( false );
        }
        backend().reset();
    }

    private boolean _currentThreadIsNotAuthorized() { return !this.equals(_INSTANCES.get()); }

    public String toString() {
        return "Neureka[" +
                    "settings=" + _settings + "," +
                    "utility="  + _utility  + "," +
                    "backend="  + _backend  +
                "]";
    }

    /**
     * @return A context object which is expected to host all the tensor operations...
     */
    public BackendContext getBackend() { return this.backend(); }

    /**
     *  Use this method to attach a backend context (for operations)
     *  to this thread local library context.
     *
     * @param backendContext The {@link BackendContext} which should be set for this thread local library context.
     */
    public void setBackend( BackendContext backendContext ) { _backend = backendContext; }

    /**
     *  This class hosts the settings of the {@link Neureka} instance which will be used throughout the library.
     */
    public class Settings
    {
        private final Debug    _debug;
        private final AutoGrad _autograd;
        private final View     _view;
        private final NDim     _nDim;
        private final DType    _dTpe;

        private boolean _isLocked = false;

        private Settings() {
            _debug    = new Debug();
            _autograd = new AutoGrad();
            _view     = new View();
            _nDim = new NDim();
            _dTpe = new DType();
        }

        private boolean notModifiable() {
            if ( _isLocked || _currentThreadIsNotAuthorized() ) {
                if ( _isLocked )
                    _LOG.error("Cannot modify settings! They are locked.");
                else
                    _LOG.error("Cannot modify settings! Current thread not authorized.");
                return true;
            }
            else return false;
        }
        
        public Debug debug() { return _debug; }

        /**
         *  This allows you to configure Neureka using a Groovy DSL.
         */
        public Debug debug(Object closure) {
            SettingsLoader.tryGroovyClosureOn(closure, _debug);
            return _debug;
        }

        public AutoGrad autograd() { return _autograd; }

        /**
         *  This allows you to configure Neureka using a Groovy DSL.
         */
        public AutoGrad autograd( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _autograd );
            return _autograd;
        }

        public View view() { return _view; }

        public View view( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _view );
            return _view;
        }

        public NDim ndim() { return _nDim; }

        /**
         *  This allows you to configure Neureka using a Groovy DSL.
         */
        public NDim ndim( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _nDim);
            return _nDim;
        }

        public DType dtype() { return _dTpe; }

        /**
         *  This allows you to configure Neureka using a Groovy DSL.
         */
        public DType dtype( Object closure ) {
            SettingsLoader.tryGroovyClosureOn( closure, _dTpe);
            return _dTpe;
        }

        /**
         *  Locked settings can only be read but not written to.
         *  Trying to write to a locked {@link Settings} instance will not have an effect.
         *  The attempt, however, will be logged.
         */
        public boolean isLocked() { return  _isLocked; }

        /**
         *  Can be used to lock or unlock the settings of the current thread-local {@link Neureka} instance.
         *  Locked settings can only be read but not written to.
         *  Trying to write to a locked {@link Settings} instance will not have an effect.
         *  The attempt, however, will be logged.
         */
        public void setIsLocked( boolean locked ) { _isLocked = locked; }

        public String toString() {
            return "Neureka.Settings[" +
                        "debug="    + _debug    + "," +
                        "autograd=" + _autograd + "," +
                        "view="     + _view     + "," +
                        "ndim="     + _nDim + "," +
                        "dtype="    + _dTpe + "," +
                        "isLocked=" + this.isLocked() +
                    "]";
        }

        
        public class Debug
        {
            private boolean _isKeepingDerivativeTargetPayloads = false;
            private boolean _isDeletingIntermediateTensors = true;

            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation,
             * which means it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             * <p>
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            public boolean isKeepingDerivativeTargetPayloads() { return _isKeepingDerivativeTargetPayloads; }

            /**
             * Every derivative is calculated with respect to some graph node.
             * Graph nodes contain payload tensors.
             * A tensor might not always be used for backpropagation,
             * which means it will be deleted if possible.
             * Targeted tensors are either leave tensors (They require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * In this case:
             * If the tensor is not needed for backpropagation it will be deleted.
             * The graph node will dereference the tensor either way.
             * <p>
             * The flag determines this behavior with respect to target nodes.
             * It is used in the test suit to validate that the right tensors were calculated.
             * This flag should not be modified in production! (memory leak)
             */
            public void setIsKeepingDerivativeTargetPayloads( boolean keep ) {
                if ( notModifiable() ) return;
                _isKeepingDerivativeTargetPayloads = keep;
            }

            /**
             * {@link neureka.math.Function} instances will produce hidden intermediate results
             * when executing an array of inputs.
             * These tensors might not always be used for backpropagation,
             * which means they will be deleted if possible.
             * Tensors are not deleted of they are leave tensors (They are created by the user or require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * This flag should not be modified in production! (memory leak)
             */
            public boolean isDeletingIntermediateTensors() { return _isDeletingIntermediateTensors; }

            /**
             * {@link neureka.math.Function} instances will produce hidden intermediate results
             * when executing an array of inputs.
             * These tensors might not always be used for backpropagation,
             * which means they will be deleted if possible.
             * Tensors are not deleted of they are leave tensors (They are created by the user or require gradients)
             * or they are angle points between forward- and reverse-mode-AutoDiff!
             * This flag should not be modified in production! (memory leak)
             */
            public void setIsDeletingIntermediateTensors( boolean delete ) {
                if ( notModifiable() ) return;
                _isDeletingIntermediateTensors = delete;
            }

            public String toString() {
                return "Neureka.Settings.Debug[" +
                            "isKeepingDerivativeTargetPayloads=" + this.isKeepingDerivativeTargetPayloads() +
                        "]";
            }
        }

        /**
         * This class contains settings which are related to the automatic differentiation of tensors.
         */
        public class AutoGrad
        {
            private boolean _isPreventingInlineOperations = true;
            private boolean _isRetainingPendingErrorForJITProp = false;
            private boolean _isApplyingGradientWhenTensorIsUsed = false;
            private boolean _isApplyingGradientWhenRequested = false;

            /**
             *  Inline operations are operations where the data of a tensor passed into an operation
             *  is being modified.
             *  Usually the result of an operation is stored inside a new tensor.
             *  Use this flag to detect if an operation is an inline operation.
             */
            public boolean isPreventingInlineOperations() { return _isPreventingInlineOperations; }

            /**
             *  Inline operations are operations where the data of a tensor passed into an operation
             *  is being modified.
             *  Usually the result of an operation is stored inside a new tensor.
             *  Use this flag to detect if an operation is an inline operation.
             */
            public void setIsPreventingInlineOperations( boolean prevent ) {
                if ( _isLocked || _currentThreadIsNotAuthorized() ) return;
                _isPreventingInlineOperations = prevent;
            }

            /**
             *  This flag enables an optimization technique which only propagates error values to
             *  gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them
             *  at divergent differentiation paths within the computation graph.<br>
             *  If the flag is set to true <br>
             *  then error values will accumulate at such junction nodes.
             *  This technique however uses more memory but will
             *  improve performance for some networks substantially.
             *  The technique is termed JIT-Propagation.
             */
            public boolean isRetainingPendingErrorForJITProp() { return _isRetainingPendingErrorForJITProp; }

            /**
             *  This flag enables an optimization technique which only propagates error values to
             *  gradients if needed by a tensor (the tensor is used again) and otherwise accumulate them
             *  at divergent differentiation paths within the computation graph.<br>
             *  If the flag is set to true <br>
             *  then error values will accumulate at such junction nodes.
             *  This technique however uses more memory but will
             *  improve performance for some networks substantially.
             *  The technique is termed JIT-Propagation.
             */
            public void setIsRetainingPendingErrorForJITProp( boolean retain ) {
                if ( _isLocked || _currentThreadIsNotAuthorized() ) return;
                _isRetainingPendingErrorForJITProp = retain;
            }

            /**
             *  Gradients will automatically be applied (or JITed) to tensors as soon as
             *  they are being used for calculation ({@link neureka.autograd.GraphNode} instantiation).
             *  This feature works well with JIT-Propagation.
             */
            public boolean isApplyingGradientWhenTensorIsUsed() { return _isApplyingGradientWhenTensorIsUsed; }

            /**
             *  Gradients will automatically be applied (or JITed) to tensors as soon as
             *  they are being used for calculation ({@link neureka.autograd.GraphNode} instantiation).
             *  This feature works well with JIT-Propagation.
             *
             * @param apply The flag determining if gradients should be applied when their tensors are used.
             */
            public void setIsApplyingGradientWhenTensorIsUsed( boolean apply ) {
                if ( _isLocked || _currentThreadIsNotAuthorized() ) return;
                _isApplyingGradientWhenTensorIsUsed = apply;
            }

            /**
             *  Gradients will only be applied if requested.
             *  Usually this happens immediately, however
             *  if the flag <i>'applyGradientWhenTensorIsUsed'</i> is set
             *  to true, then the tensor will only be updated by its
             *  gradient if requested AND the tensor is used for calculation!
             *  ({@link neureka.autograd.GraphNode} instantiation).  <br> <br>
             *
             *  This flag works alongside two other autograd features which can be enabled by flipping the feature flags <br>
             *  <i>'isApplyingGradientWhenRequested'</i> and <i>'isApplyingGradientWhenTensorIsUsed'</i><br>
             *  As the first flag name suggests gradients will be applied to their tensors when it is set to true,
             *  however this will only happen when the second flag is set to true as well, because otherwise gradients
             *  wouldn't be applied to their tensors automatically in the first place... <br>
             *  <br>
             *  Setting both flags to true will inhibit the effect of the second setting <i>'isApplyingGradientWhenTensorIsUsed'</i>
             *  unless a form of "permission" is being signaled to the autograd system.
             *  This signal comes in the form of a "request" flag which marks a tensor as <b>allowed to
             *  be updated by its gradient</b>. This request can be dispatched to a {@link Tsr}
             *  by setting {@link Tsr#setGradientApplyRequested(boolean)} to {@code true}.<br>
             *
             * @return The truth value determining if gradients should be applied upon request.
             **/
            public boolean isApplyingGradientWhenRequested() { return _isApplyingGradientWhenRequested; }

            /**
             * Gradients will only be applied if requested.
             * Usually this happens immediately, however
             * if the flag <i>'applyGradientWhenTensorIsUsed'</i> is set
             * to true, then the tensor will only be updated by its
             * gradient if requested AND the tensor is used for calculation!
             * ({@link neureka.autograd.GraphNode} instantiation).
             */
            public void setIsApplyingGradientWhenRequested(boolean apply) {
                if ( notModifiable() ) return;
                _isApplyingGradientWhenRequested = apply;
            }

            public String toString() {
                return "Neureka.Settings.AutoGrad[" +
                            "isPreventingInlineOperations=" + this.isPreventingInlineOperations() + "," +
                            "isRetainingPendingErrorForJITProp=" + this.isRetainingPendingErrorForJITProp() + "," +
                            "isApplyingGradientWhenTensorIsUsed=" + this.isApplyingGradientWhenTensorIsUsed() + "," +
                            "isApplyingGradientWhenRequested=" + this.isApplyingGradientWhenRequested() +
                        "]";
            }
        }

        /**
         *  Settings for configuring how objects should be converted to {@link String} representations.
         */
        public class View
        {
            private final NDPrintSettings _settings;

            View() { _settings = new NDPrintSettings(Settings.this::notModifiable); }

            /**
             *  Settings for configuring how tensors should be converted to {@link String} representations.
             */
            public NDPrintSettings getNDPrintSettings() { return _settings; }

            /**
             *  This allows you to provide a lambda to configure how tensors should be
             *  converted to {@link String} instances.
             *  The provided {@link Consumer} will receive a {@link NDPrintSettings} instance
             *  which allows you to change various settings with the help of method chaining.
             *
             * @param should A consumer of the {@link NDPrintSettings} ready to be configured.
             */
            public void ndArrays( Consumer<NDPrintSettings> should ) { should.accept(_settings); }

            public String toString() {
                return "Neureka.Settings.View[" +
                            "ndPrintSettings=" + this.getNDPrintSettings() +
                        "]";
            }
        }

        /**
         *  Settings for configuring the access pattern of nd-arrays/tensors.
         */
        public class NDim
        {
            /**
             *  The {@link SlicedNDConfiguration}
             *  class stores shape, translation... as cached int arrays.
             *  Disabling this flag allows for custom 1D, 2D, 3D classes to be loaded. (Improves memory locality)
             */
            private boolean _isOnlyUsingDefaultNDConfiguration = false;

            /**
             * This flag determines which {@link neureka.ndim.config.NDConfiguration} implementations
             * should be used for nd-arrays/tensors.
             * If this flag is set to true, then the less performant general purpose {@link neureka.ndim.config.NDConfiguration}
             * will be used for all nd-arrays/tensors.
             *
             * @return The truth value determining if only the default {@link SlicedNDConfiguration} should be used.
             */
            public boolean isOnlyUsingDefaultNDConfiguration() { return _isOnlyUsingDefaultNDConfiguration; }

            /**
             * Setting this flag determines which {@link neureka.ndim.config.NDConfiguration} implementations
             * should be used for nd-arrays/tensors.
             * If this flag is set to true, then the less performant general purpose {@link neureka.ndim.config.NDConfiguration}
             * will be used for all nd-arrays/tensors.
             *
             * @param enabled The truth value determining if only the default {@link SlicedNDConfiguration} should be used.
             */
            public void setIsOnlyUsingDefaultNDConfiguration( boolean enabled ) {
                if ( notModifiable() ) return;
                _isOnlyUsingDefaultNDConfiguration = enabled;
            }

            public String toString() {
                return "Neureka.Settings.NDim[" +
                              "isOnlyUsingDefaultNDConfiguration=" + this.isOnlyUsingDefaultNDConfiguration() +
                        "]";
            }
        }

        
        public class DType {

            private Class<?> _defaultDataTypeClass = F64.class;

            private boolean _isAutoConvertingExternalDataToJVMTypes = true;

            /**
             *  The default data type is not relevant most of the time.
             *  However, if a tensor is being constructed without providing a type class,
             *  then this property will be used.
             */
            public Class<?> getDefaultDataTypeClass() { return _defaultDataTypeClass; }

            public DataType<?> getDefaultDataType() {
                return DataType.of( _defaultDataTypeClass );
            }

            /**
             *  The default data type is not relevant most of the time.
             *  However, if a tensor is being constructed without providing a type class,
             *  then this property will be used.
             */
            public void setDefaultDataTypeClass( Class<?> dtype ) {
                if ( notModifiable() ) return;
                _defaultDataTypeClass = dtype;
            }

            /**
             *  This flag will determine if foreign data types will be converted into the next best fit (in terms of bits)
             *  or if it should be converted into something that does not mess with the representation of the data.
             *  For example an unsigned int can be converted bit-wise into a JVM int, or
             *  it could be converted to a JVM long type in order to be compatible with JVM operations...
             */
            public boolean getIsAutoConvertingExternalDataToJVMTypes() { return _isAutoConvertingExternalDataToJVMTypes; }

            /**
             *  This flag will determine if foreign data types will be converted into the next best fit (in terms of bits)
             *  or if it should be converted into something that does not mess with the representation of the data.
             *  For example an unsigned int can be converted bit-wise into a JVM int, or
             *  it could be converted to a JVM long type in order to be compatible with JVM operations...
             */
            public void setIsAutoConvertingExternalDataToJVMTypes( boolean autoConvert ) {
                if ( notModifiable() ) return;
                _isAutoConvertingExternalDataToJVMTypes = autoConvert;
            }

            public String toString() {
                return "Neureka.Settings.DType[" +
                            "defaultDataTypeClass=" + this.getDefaultDataTypeClass() + "," +
                            "isAutoConvertingExternalDataToJVMTypes=" + _isAutoConvertingExternalDataToJVMTypes +
                        "]";
            }
        }

    }

    public static class Utility
    {
        /**
         * Helper method which reads the file with the given name and returns
         * the contents of this file as a String. Will exit the application
         * if the file can not be read.
         *
         * @param path The path to the jar resource.
         * @return The contents of the file
         */
        public String readResource( String path ) {
            InputStream stream = getClass().getClassLoader().getResourceAsStream( path );
            if ( stream == null )
                throw new IllegalStateException(
                        "Failed to create InputStream for resource path '"+path+"'."
                    );
            try {
                BufferedReader br = new BufferedReader(new InputStreamReader( stream ));
                StringBuilder sb = new StringBuilder();
                String line = "";
                while ( line != null ) {
                    line = br.readLine();
                    if ( line != null ) sb.append( line ).append( "\n" );
                }
                return sb.toString();
            } catch ( IOException e ) {
                _LOG.error("Failed loading library resource at '"+path+"'!");
                return "";
            }
        }

        /**
         *  This method checks if a class addressed by its name has been loaded into the runtime.
         *
         * @param className The class whose presents ought to be checked.
         * @return The truth value determining if the class is present or not.
         */
        static boolean isPresent( String className, Supplier<String> tip ) {
            boolean found = false;
            String cause = " unknown ";
            try {
                Class.forName( className );
                found = true;
            } catch ( Throwable ex ) {// Class or one of its dependencies is not present...
                cause = ex.getMessage();
            } finally {
                String tipMessage = tip.get().replace("\n", "\n    "+"     ").trim();
                if ( !found )
                    _LOG.debug(
                        "Neureka:\n" +
                        "    info: Failed to load class '" + className + "'!" + "\n" +
                        "    cause: " + cause + "\n" +
                        "    tip: " + tipMessage + "\n"
                    );
            }
            return found;
        }

    }

}
