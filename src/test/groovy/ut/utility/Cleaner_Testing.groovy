package ut.utility

import neureka.Tsr
import neureka.devices.DeviceCleaner
import spock.lang.Specification
import spock.lang.Title
import spock.lang.Narrative
import testutility.Sleep

@Title("How Neureka Cleans Up")
@Narrative ('''

    Under the hood 
    Neureka deals whith large arrays of
    data, which are often times 
    native data arrays requiring explicit
    memory freeing!
    This freeing of memory can happen at any time
    during the livetime of a nd-array, however
    it should happen at least up until the nd-arra/tensor
    objects representing their referenced data arrays become
    eligible for garbage collection.
    This specification ensures that the custom garbage
    cleaner implementation used by Neureka fulfills this role 

''')
class Cleaner_Testing extends Specification
{

    def 'The DeviceCleaner triggers registered cleaner actions when things are eligible for GC.'(){

        given :
            var cleaner = DeviceCleaner.getInstance()
            var refCount = 10
            var r1 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r2 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r3 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r4 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r5 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r6 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r7 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r8 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r9 =  Tsr.ofDoubles().withShape( 2, 4 ).all(0)
            var r10 = Tsr.ofDoubles().withShape( 2, 4 ).all(0)

            cleaner.register( r1, {refCount--} )
            cleaner.register( r2, {refCount--} )
            cleaner.register( r3, {refCount--} )
            cleaner.register( r4, {refCount--} )
            cleaner.register( r5, {refCount--} )
            cleaner.register( r6, {refCount--} )
            cleaner.register( r7, {refCount--} )
            cleaner.register( r8, {refCount--} )
            cleaner.register( r9, {refCount--} )
            cleaner.register( r10,{refCount--} )

        when :
            r1 = null
            r3 = null
            System.gc()

        then :
            Sleep.until(700, { refCount == 8 && cleaner._registered == 8 })
            r1  == null
            r2  != null
            r3  == null
            r4  != null
            r5  != null
            r6  != null
            r7  != null
            r8  != null
            r9  != null
            r10 != null

        when :
            r2 = null
            r4 = null
            System.gc()

        then :
            Sleep.until(750, { refCount == 6 && cleaner._registered == 6 })
            r1  == null
            r2  == null
            r3  == null
            r4  == null
            r5  != null
            r6  != null
            r7  != null
            r8  != null
            r9  != null
            r10 != null
    }


}
