already_marked = set(attendance[attendance['Date'] == today_date]['Id'].astype(str).values) if not attendance.empty else set()
