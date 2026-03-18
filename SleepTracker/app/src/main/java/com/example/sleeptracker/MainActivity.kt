package com.example.sleeptracker

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import com.example.sleeptracker.ui.screens.DashboardScreen
import com.example.sleeptracker.ui.screens.TrackingSheet
import com.example.sleeptracker.ui.theme.SleepTrackerTheme

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SleepTrackerTheme {
                var showTrackingSheet by remember { mutableStateOf(false) }
                val sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = false)

                DashboardScreen(
                    context = applicationContext,
                    onTrackingClick = { showTrackingSheet = true }
                )

                if (showTrackingSheet) {
                    ModalBottomSheet(
                        onDismissRequest = { showTrackingSheet = false },
                        sheetState = sheetState,
                        containerColor = com.example.sleeptracker.ui.theme.DarkBackground,
                        scrimColor = androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.3f),
                        modifier = Modifier.fillMaxSize()
                    ) {
                        TrackingSheet(
                            context = applicationContext,
                            onDismiss = { showTrackingSheet = false }
                        )
                    }
                }
            }
        }
    }
}
