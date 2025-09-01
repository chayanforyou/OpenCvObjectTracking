package io.github.chayanforyou.tracking

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Paint
import android.graphics.Point
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.Size
import android.view.MotionEvent
import android.view.Surface
import android.view.TextureView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.Rect2d
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.tracking.legacy_Tracker
import org.opencv.tracking.legacy_TrackerCSRT
import org.opencv.tracking.legacy_TrackerKCF
import org.opencv.tracking.legacy_TrackerMIL
import org.opencv.tracking.legacy_TrackerMOSSE
import org.opencv.tracking.legacy_TrackerMedianFlow
import org.opencv.tracking.legacy_TrackerTLD
import kotlin.math.max
import kotlin.math.min

enum class Drawing { DRAWING, TRACKING, CLEAR }

class MainActivity : AppCompatActivity() {

    private lateinit var textureView: TextureView
    private lateinit var trackingOverlay: OverlayView

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var previewRequestBuilder: CaptureRequest.Builder? = null
    private var imageGrab: Mat? = null
    private var imageReader: ImageReader? = null
    private var tracker: legacy_Tracker? = null
    private var drawing = Drawing.DRAWING
    private val camResolution = Size(1280, 720)
    private val points = arrayOf(Point(0, 0), Point(0, 0))
    private var processing = false
    private var targetLocked = false
    private var handler = Handler(Looper.getMainLooper())
    private var selectedTracker = "TrackerKCF"  // OpenCV tracking algorithm

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                openCamera()
            } else {
                Toast.makeText(this, "Sorry!!!, you can't use this app without granting permission", Toast.LENGTH_LONG).show()
            }
        }

    private val textureListener = object : TextureView.SurfaceTextureListener {
        override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
            openCamera()
        }
        override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}
        override fun onSurfaceTextureDestroyed(surface: SurfaceTexture) = false
        override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!OpenCVLoader.initLocal()) {
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show()
            return
        }
        setContentView(R.layout.activity_main)
        textureView = findViewById(R.id.texture)
        trackingOverlay = findViewById(R.id.tracking_overlay)
        textureView.surfaceTextureListener = textureListener
    }

    override fun onDestroy() {
        closeCamera()
        super.onDestroy()
    }

    private fun openCamera() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }

        val manager = getSystemService(CAMERA_SERVICE) as CameraManager
        try {
            val cameraId = manager.cameraIdList.first()
            manager.openCamera(cameraId, stateCallback, null)
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private fun closeCamera() {
        captureSession?.close()
        captureSession = null
        cameraDevice?.close()
        cameraDevice = null
        imageReader?.close()
        imageReader = null
    }

    private val stateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            cameraDevice = camera
            createCameraPreview()
        }

        override fun onDisconnected(camera: CameraDevice) {
            cameraDevice?.close()
        }

        override fun onError(camera: CameraDevice, error: Int) {
            cameraDevice?.close()
            cameraDevice = null
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private fun createCameraPreview() {
        try {
            val texture = textureView.surfaceTexture
            texture?.setDefaultBufferSize(camResolution.width, camResolution.height)
            val surface = Surface(texture)
            previewRequestBuilder = cameraDevice?.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)?.apply {
                addTarget(surface)
            }
            imageReader = ImageReader.newInstance(camResolution.width, camResolution.height, ImageFormat.JPEG, 2).apply {
                setOnImageAvailableListener(onImageAvailableListener, handler)
            }
            previewRequestBuilder?.addTarget(imageReader!!.surface)
            cameraDevice?.createCaptureSession(
                listOf(surface, imageReader!!.surface),
                object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        if (cameraDevice == null) return
                        captureSession = session
                        previewRequestBuilder?.set(
                            CaptureRequest.CONTROL_AF_MODE,
                            CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE
                        )
                        previewRequestBuilder?.set(
                            CaptureRequest.CONTROL_AE_MODE,
                            CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH
                        )
                        try {
                            session.setRepeatingRequest(
                                previewRequestBuilder!!.build(),
                                null,
                                handler
                            )
                        } catch (e: CameraAccessException) {
                            e.printStackTrace()
                        }
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        Toast.makeText(this@MainActivity, "Configuration change", Toast.LENGTH_SHORT).show()
                    }
                },
                null
            )

            trackingOverlay.addCallback(object : OverlayView.DrawCallback {
                override fun drawCallback(canvas: Canvas) {
                    if (drawing != Drawing.CLEAR) {
                        val paint = Paint().apply {
                            color = Color.BLUE
                            strokeWidth = 10f
                            style = Paint.Style.STROKE
                        }
                        canvas.drawRect(
                            points[0].x.toFloat(),
                            points[0].y.toFloat(),
                            points[1].x.toFloat(),
                            points[1].y.toFloat(),
                            paint
                        )
                        if (drawing == Drawing.TRACKING) {
                            paint.color = Color.GREEN
                            canvas.drawLine(
                                (points[0].x + points[1].x) / 2f,
                                0f,
                                (points[0].x + points[1].x) / 2f,
                                trackingOverlay.height.toFloat(),
                                paint
                            )
                            canvas.drawLine(
                                0f,
                                (points[0].y + points[1].y) / 2f,
                                trackingOverlay.width.toFloat(),
                                (points[0].y + points[1].y) / 2f,
                                paint
                            )
                        }
                    }
                }
            })

            trackingOverlay.setOnTouchListener { _, event ->
                val x = event.x.toInt()
                val y = event.y.toInt()
                when (event.action and MotionEvent.ACTION_MASK) {
                    MotionEvent.ACTION_DOWN -> if (!targetLocked) {
                        drawing = Drawing.DRAWING
                        points[0].set(x, y)
                        points[1].set(x, y)
                        trackingOverlay.invalidate()
                    }
                    MotionEvent.ACTION_MOVE -> if (!targetLocked) {
                        points[1].set(x, y)
                        trackingOverlay.invalidate()
                    }
                    MotionEvent.ACTION_UP -> if (!targetLocked) {
                        drawing = Drawing.CLEAR
                        trackingOverlay.invalidate()
                    }
                    MotionEvent.ACTION_POINTER_DOWN -> {
                        targetLocked = !targetLocked
                        Toast.makeText(this@MainActivity, "Target ${if (targetLocked) "LOCKED" else "UNLOCKED"}", Toast.LENGTH_SHORT).show()
                        drawing = Drawing.DRAWING
                        trackingOverlay.invalidate()
                    }
                }
                true
            }
        } catch (e: CameraAccessException) {
            e.printStackTrace()
        }
    }

    private val onImageAvailableListener = ImageReader.OnImageAvailableListener { reader ->
        val image = reader.acquireLatestImage() ?: return@OnImageAvailableListener
        if (processing) {
            image.close()
            return@OnImageAvailableListener
        }
        processing = true

        if (targetLocked) {
            val bb = image.planes[0].buffer
            val data = ByteArray(bb.remaining()).also { bb.get(it) }
            imageGrab = Imgcodecs.imdecode(MatOfByte(*data), Imgcodecs.IMREAD_UNCHANGED)
            Core.transpose(imageGrab, imageGrab)
            Core.flip(imageGrab, imageGrab, 1)
            Imgproc.resize(imageGrab, imageGrab, org.opencv.core.Size(240.0, 320.0))
        }
        image.close()
        processing()
    }

    private fun processing() {
        if (targetLocked && imageGrab != null) {
            if (drawing == Drawing.DRAWING) {
                val minX = (min(points[0].x.toFloat(), points[1].x.toFloat()) / trackingOverlay.width * imageGrab!!.cols()).toInt()
                val minY = (min(points[0].y.toFloat(), points[1].y.toFloat())/ trackingOverlay.height * imageGrab!!.rows()).toInt()
                val maxX = (max(points[0].x.toFloat(), points[1].x.toFloat())/ trackingOverlay.width * imageGrab!!.cols()).toInt()
                val maxY = (max(points[0].y.toFloat(), points[1].y.toFloat()) / trackingOverlay.height * imageGrab!!.rows()).toInt()

                val initRectangle = Rect2d(
                    minX.toDouble(),
                    minY.toDouble(),
                    (maxX - minX).toDouble(),
                    (maxY - minY).toDouble()
                )
                val imageGrabInit = Mat().also { imageGrab?.copyTo(it) }

                tracker = when (selectedTracker) {
                    "TrackerMedianFlow" -> legacy_TrackerMedianFlow.create()
                    "TrackerCSRT" -> legacy_TrackerCSRT.create()
                    "TrackerKCF" -> legacy_TrackerKCF.create()
                    "TrackerMOSSE" -> legacy_TrackerMOSSE.create()
                    "TrackerTLD" -> legacy_TrackerTLD.create()
                    "TrackerMIL" -> legacy_TrackerMIL.create()
                    else -> legacy_TrackerMedianFlow.create()
                }
                tracker!!.init(imageGrabInit, initRectangle)
                drawing = Drawing.TRACKING
            } else {
                val trackingRectangle = Rect2d(0.0, 0.0, 1.0, 1.0)
                tracker?.update(imageGrab!!, trackingRectangle)
                points[0].x = (trackingRectangle.x * trackingOverlay.width / imageGrab!!.cols()).toInt()
                points[0].y = (trackingRectangle.y * trackingOverlay.height / imageGrab!!.rows()).toInt()
                points[1].x = points[0].x + (trackingRectangle.width * trackingOverlay.width / imageGrab!!.cols()).toInt()
                points[1].y = points[0].y + (trackingRectangle.height * trackingOverlay.height / imageGrab!!.rows()).toInt()
                trackingOverlay.invalidate()
            }
        } else {
            tracker?.clear()
            tracker = null
        }
        processing = false
    }
}