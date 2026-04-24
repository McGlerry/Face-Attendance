document.addEventListener('DOMContentLoaded', function() {
    console.log('Reports JS loaded successfully');

    // Stub notification if not defined
    window.showNotification = window.showNotification || function(msg, type) {
        console.log(`Notification [${type}]: ${msg}`);
        // Could add real toast here
    };

    // Set default dates
    const today = new Date();
    const startDate = new Date(today);
    startDate.setDate(today.getDate() - 7);
    
    const startDateEl = document.getElementById('start_date');
    const endDateEl = document.getElementById('end_date');
    const reportForm = document.getElementById('reportForm');
    const reportResults = document.getElementById('reportResults');
    const exportCSV = document.getElementById('exportCSV');
    const exportDOCX = document.getElementById('exportDOCX');
    const exportPDF = document.getElementById('exportPDF');

    if (startDateEl && endDateEl) {
startDate.toISOString().split('T')[0] || '';
today.toISOString().split('T')[0] || '';
    }

    if (exportCSV) exportCSV.addEventListener('click', function() { exportReport('csv'); });
    if (exportDOCX) exportDOCX.addEventListener('click', function() { exportReport('docx'); });
    if (exportPDF) exportPDF.addEventListener('click', function() { exportReport('pdf'); });

    if (reportForm) {
        reportForm.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Report form submitted');
            
            if (!reportResults) {
                console.error('reportResults element not found');
                return;
            }

            // Show loading
            reportResults.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Generating report...</p></div>';

            const formData = new FormData(reportForm);
            
            fetch('/api/generate_report', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('Report data received:', data);
                if(data.success) {
            displayReportResults(data.records, data.count);
            if (exportCSV) exportCSV.disabled = (data.count === 0);
            if (exportDOCX) exportDOCX.disabled = (data.count === 0);
            if (exportPDF) exportPDF.disabled = (data.count === 0);
                } else {
                    reportResults.innerHTML = '<div class="alert alert-danger">Failed to generate report: ' + (data.error || 'Unknown error') + '</div>';
                    showNotification('Failed to generate report: ' + data.error, 'danger');
                }
            })
            .catch(error => {
                console.error('Report fetch error:', error);
                reportResults.innerHTML = '<div class="alert alert-danger">Network error. Please try again.</div>';
                showNotification('An error occurred during report generation.', 'danger');
            });
        });
    }
});

function getStatusClass(status) {
    if (status === 'absent') {
        return 'bg-danger';
    } else if (status === 'late') {
        return 'bg-warning';
    } else {
        return 'bg-success';
    }
}

function displayReportResults(records, count) {
    let html = `
        <div class="mb-3">
            <span class="badge bg-primary">${count} Records Found</span>
        </div>
    `;

    if(records.length > 0) {
        // Group records by class
        const groupedByClass = {};
        records.forEach(record => {
            if (!groupedByClass[record.class_name]) {
                groupedByClass[record.class_name] = [];
            }
            groupedByClass[record.class_name].push(record);
        });

        // Sort classes
        const sortedClasses = Object.keys(groupedByClass).sort();

        sortedClasses.forEach(className => {
            const classRecords = groupedByClass[className];
            
            html += `
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-chalkboard me-2"></i>${className} 
                            <span class="badge bg-secondary ms-2">${classRecords.length} Records</span>
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-hover">
                                <thead>
                                    <tr>
                                        <th>Student ID</th>
                                        <th>Student Name</th>
                                        <th>Timestamp</th>
                                        <th>Confidence</th>
                                        <th>Status</th>
                                        <th>Match Type</th>
                                    </tr>
                                </thead>
                                <tbody>
            `;

            // Sort records by student name
            classRecords.sort((a, b) => a.name.localeCompare(b.name));

            classRecords.forEach(record => {
                const timestamp = record.timestamp ? new Date(record.timestamp).toLocaleString() : record.date;
                html += `
                    <tr>
                        <td><strong>${record.student_id}</strong></td>
                        <td>${record.name}</td>
                        <td>${timestamp}</td>
                        <td>${(record.confidence_score || 0).toFixed(3)}</td>
                        <td><span class="badge ${getStatusClass(record.status)}">${record.status || 'Present'}</span></td>
                        <td>${record.match_type}</td>
                    </tr>
                `;
            });

            html += `
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
        });
    } else {
        html += `
            <div class="text-center py-4">
                <i class="fas fa-inbox fa-2x text-muted mb-3"></i>
                <h6>No Records Found</h6>
                <p class="text-muted">No attendance records found for the selected criteria.</p>
            </div>
        `;
    }

    document.getElementById('reportResults').innerHTML = html;
}

function exportReport(format) {
    const start_date = document.getElementById('start_date').value;
    const end_date = document.getElementById('end_date').value;
    const class_id = document.getElementById('class_id').value;

    const formData = new FormData();
    formData.append('start_date', start_date);
    formData.append('end_date', end_date);
    formData.append('class_id', class_id);

    let url = '';
    if (format === 'csv') {
        url = '/api/export_csv';
    } else if (format === 'pdf') {
        url = '/api/export_pdf';
    } else if (format === 'docx') {
        url = '/api/export_docx';
    } else {
        showNotification('Invalid export format.', 'danger');
        return;
    }

    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (response.ok) {
            // For direct download, get the blob and create download link
            return response.blob().then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                
                // PERFECTLY SAFE: Backend sets Content-Disposition header, static name works
                    const filename = `attendance_report_${format.toUpperCase()}_${new Date().toISOString().slice(0,10)}.${format}`;


                a.download = filename;
                
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showNotification(`${format.toUpperCase()} report downloaded successfully!`, 'success');
            });
        } else {
            return response.json().then(data => {
                throw new Error(data.message || 'Export failed');
            });
        }
    })
    .catch(error => {
        console.error(`Error exporting ${format.toUpperCase()}:`, error);
        showNotification(`Failed to export ${format.toUpperCase()}: ${error.message}`, 'danger');
    });
}

